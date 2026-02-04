"""
Ollama API Router for Claude Code
Handles rate limiting by rotating through multiple Ollama instances
Makes Ollama compatible with Claude Code and Anthropic SDK
"""

import asyncio
import json
import logging
import os
import uuid
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
from peewee import SqliteDatabase, Model, CharField, IntegerField, DateTimeField, BooleanField

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Request Deduplication - Prevent duplicate requests to Ollama
# ============================================================================

class RequestCache:
    """Cache recent requests to prevent duplicates within a time window"""
    def __init__(self, ttl_seconds=2):
        self.cache = {}  # {request_hash: (timestamp, result)}
        self.ttl = ttl_seconds
    
    def get_hash(self, data: dict) -> str:
        """Generate hash of request data"""
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def is_duplicate(self, data: dict) -> bool:
        """Check if this request was recently processed"""
        request_hash = self.get_hash(data)
        if request_hash in self.cache:
            timestamp, _ = self.cache[request_hash]
            if datetime.now().timestamp() - timestamp < self.ttl:
                return True
        return False
    
    def add(self, data: dict):
        """Record this request"""
        request_hash = self.get_hash(data)
        self.cache[request_hash] = (datetime.now().timestamp(), None)
        # Cleanup old entries
        current_time = datetime.now().timestamp()
        self.cache = {k: v for k, v in self.cache.items() 
                      if current_time - v[0] < self.ttl * 2}

request_cache = RequestCache(ttl_seconds=2)

# ============================================================================
# Database Setup with Peewee
# ============================================================================

# Create data directory if it doesn't exist
os.makedirs('/app/data', exist_ok=True)

db = SqliteDatabase('/app/data/ollama_metrics.db')

class TokenMetrics(Model):
    """Tracks comprehensive metrics per account"""
    account_name = CharField(unique=True)
    tokens_uploaded = IntegerField(default=0)
    tokens_downloaded = IntegerField(default=0)
    requests_made = IntegerField(default=0)
    tool_calls = IntegerField(default=0)
    rate_limit_count = IntegerField(default=0)
    last_error = CharField(default="")
    is_rate_limited = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    
    class Meta:
        database = db
        table_name = 'token_metrics'

def init_db():
    """Initialize database and create tables"""
    db.connect()
    db.create_tables([TokenMetrics], safe=True)
    logger.info("Database initialized")

# ============================================================================
# Model Mapping - Convert all Claude model names to single Ollama Cloud model
# ============================================================================

DEFAULT_MODEL = "glm-4.7:cloud"

def map_model_name(claude_model: str) -> str:
    """Map all Claude model names to the default Ollama Cloud model"""
    logger.info(f"Model mapping: {claude_model} -> {DEFAULT_MODEL}")
    return DEFAULT_MODEL

# ============================================================================
# Tool Conversion Functions - Ollama <-> Anthropic Format
# ============================================================================

def convert_anthropic_tools_to_ollama(anthropic_tools: List[dict]) -> List[dict]:
    """
    Convert Anthropic tool format to Ollama format
    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    Ollama: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    if not anthropic_tools:
        return []
    
    ollama_tools = []
    for tool in anthropic_tools:
        ollama_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            }
        }
        ollama_tools.append(ollama_tool)
    
    return ollama_tools


def convert_ollama_tool_calls_to_anthropic(ollama_tool_calls: List[dict]) -> List[dict]:
    """
    Convert Ollama tool call format to Anthropic format
    Ollama: [{"function": {"name": "...", "arguments": {...}}}]
    Anthropic: [{"type": "tool_use", "id": "...", "name": "...", "input": {...}}]
    """
    if not ollama_tool_calls:
        return []
    
    anthropic_tool_calls = []
    for tool_call in ollama_tool_calls:
        # Extract function info from Ollama format
        func = tool_call.get("function", {})
        name = func.get("name", "")
        arguments = func.get("arguments", {})
        
        # Convert to Anthropic format
        anthropic_call = {
            "type": "tool_use",
            "id": f"tool_{uuid.uuid4().hex[:12]}",
            "name": name,
            "input": arguments if isinstance(arguments, dict) else {}
        }
        anthropic_tool_calls.append(anthropic_call)
    
    return anthropic_tool_calls


def convert_anthropic_tool_results_to_ollama(messages: List[dict]) -> List[dict]:
    """
    Convert Anthropic message format with tool results to Ollama format
    Anthropic tool result: {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    Ollama tool result: {"role": "tool", "content": "...", "tool_name": "..."}
    """
    converted_messages = []
    
    for msg in messages:
        if msg.get("role") == "user":
            # Handle user messages with tool results
            content = msg.get("content", [])
            if isinstance(content, list):
                # Extract tool results and other content
                text_content = []
                tool_results = []
                
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            tool_results.append(item)
                        elif item.get("type") == "text":
                            text_content.append(item.get("text", ""))
                    else:
                        text_content.append(str(item))
                
                # Add user message if there's text content
                if text_content:
                    converted_messages.append({
                        "role": "user",
                        "content": " ".join(text_content)
                    })
                
                # Add tool result messages
                for tool_result in tool_results:
                    tool_content = tool_result.get("content", [])
                    if isinstance(tool_content, list):
                        tool_content = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in tool_content])
                    
                    converted_messages.append({
                        "role": "tool",
                        "content": str(tool_content),
                        "tool_name": tool_result.get("tool_use_id", "")
                    })
            else:
                # Simple string content
                converted_messages.append({
                    "role": "user",
                    "content": str(content)
                })
        else:
            # Pass through other message types
            converted_messages.append(msg)
    
    return converted_messages


# ============================================================================
# Configuration Models
# ============================================================================

class OllamaAccount(BaseModel):
    """Configuration for a single Ollama instance"""
    name: str
    base_url: str = "http://localhost:11434"
    max_requests_per_minute: int = 30
    api_key: Optional[str] = None  # For Ollama Cloud API
    is_cloud: bool = False  # True for ollama.com cloud API

    
class OllamaMetrics(BaseModel):
    """Tracks Ollama instance usage and rate limits"""
    name: str
    requests_made: int = 0
    tokens_uploaded: int = 0
    tokens_downloaded: int = 0
    tool_calls: int = 0
    last_rate_limit_time: Optional[datetime] = None
    is_rate_limited: bool = False
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    created_at: datetime = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.now()


# ============================================================================
# Ollama Manager
# ============================================================================

class OllamaManager:
    """Manages multiple Ollama instances and switches on rate limits"""
    
    def __init__(self, instances: List[OllamaAccount]):
        self.instances = instances
        self.metrics = {inst.name: OllamaMetrics(name=inst.name) for inst in instances}
        self.current_instance_index = 0
        self.rotation_lock = asyncio.Lock()
        
    async def get_next_available_instance(self) -> OllamaAccount:
        """Get next Ollama instance that's not rate limited"""
        async with self.rotation_lock:
            max_attempts = len(self.instances)
            attempts = 0
            
            while attempts < max_attempts:
                instance = self.instances[self.current_instance_index]
                metrics = self.metrics[instance.name]
                
                # Check if instance is rate limited and timeout has passed
                if metrics.is_rate_limited:
                    # Reset rate limit after 30 seconds
                    if metrics.last_rate_limit_time and \
                       datetime.now() - metrics.last_rate_limit_time > timedelta(seconds=30):
                        logger.info(f"Resetting rate limit for instance: {instance.name}")
                        metrics.is_rate_limited = False
                        metrics.consecutive_errors = 0
                        return instance
                    else:
                        # Instance still rate limited, try next one
                        logger.warning(f"Instance {instance.name} still rate limited, switching...")
                        self.current_instance_index = (self.current_instance_index + 1) % len(self.instances)
                        attempts += 1
                        continue
                else:
                    # Instance is available
                    return instance
            
            # If all instances are rate limited, return current anyway and let it fail
            logger.error("All instances are rate limited!")
            return self.instances[self.current_instance_index]
    
    async def mark_rate_limited(self, instance_name: str, error_msg: str = ""):
        """Mark instance as rate limited"""
        async with self.rotation_lock:
            if instance_name in self.metrics:
                metrics = self.metrics[instance_name]
                metrics.is_rate_limited = True
                metrics.last_rate_limit_time = datetime.now()
                metrics.consecutive_errors += 1
                metrics.last_error = error_msg
                logger.warning(f"Instance {instance_name} marked as rate limited (errors: {metrics.consecutive_errors})")
                
                # Persist to database
                db_record, created = TokenMetrics.get_or_create(
                    account_name=instance_name,
                    defaults={"created_at": datetime.now()}
                )
                db_record.is_rate_limited = True
                db_record.rate_limit_count += 1
                db_record.last_error = error_msg
                db_record.updated_at = datetime.now()
                db_record.save()
            
            # Switch to next instance
            self.current_instance_index = (self.current_instance_index + 1) % len(self.instances)
    
    async def record_success(self, instance_name: str, tokens_up: int = 0, tokens_down: int = 0, tool_calls_count: int = 0):
        """Record successful request with token counts and tool calls"""
        if instance_name in self.metrics:
            metrics = self.metrics[instance_name]
            metrics.requests_made += 1
            metrics.tokens_uploaded += tokens_up
            metrics.tokens_downloaded += tokens_down
            metrics.tool_calls += tool_calls_count
            metrics.consecutive_errors = 0
            
            # Persist to database
            db_record, created = TokenMetrics.get_or_create(
                account_name=instance_name,
                defaults={"created_at": datetime.now()}
            )
            db_record.tokens_uploaded += tokens_up
            db_record.tokens_downloaded += tokens_down
            db_record.requests_made += 1
            db_record.tool_calls += tool_calls_count
            db_record.is_rate_limited = metrics.is_rate_limited
            db_record.last_error = metrics.last_error or ""
            db_record.updated_at = datetime.now()
            db_record.save()
    
    def get_metrics(self) -> Dict[str, dict]:
        """Get all instance metrics"""
        return {
            name: {
                "name": metrics.name,
                "requests_made": metrics.requests_made,
                "is_rate_limited": metrics.is_rate_limited,
                "consecutive_errors": metrics.consecutive_errors,
                "last_error": metrics.last_error,
                "uptime": (datetime.now() - metrics.created_at).total_seconds(),
            }
            for name, metrics in self.metrics.items()
        }


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Ollama API Router for Claude Code",
    description="Routes Ollama API requests with automatic instance switching on rate limits",
    version="1.0.0"
)

# Global Ollama manager (will be initialized on startup)
ollama_manager: Optional[OllamaManager] = None

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token on average"""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)

def load_instances_from_file() -> List[OllamaAccount]:
    """Load Ollama instances from apikeys.txt file"""
    instances = []
    
    # Try to load from apikeys.txt
    apikeys_file = os.path.join(os.path.dirname(__file__), "apikeys.txt")
    
    if os.path.exists(apikeys_file):
        try:
            with open(apikeys_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        instance = OllamaAccount(
                            name=f"account_{i+1}",
                            base_url="https://ollama.com",
                            api_key=line,
                            max_requests_per_minute=30,
                            is_cloud=True
                        )
                        instances.append(instance)
                logger.info(f"Loaded {len(instances)} API keys from apikeys.txt")
                return instances
        except Exception as e:
            logger.error(f"Failed to load from apikeys.txt: {e}")
    
    # Fallback: check for environment variable
    instances_json = os.getenv("OLLAMA_INSTANCES")
    if instances_json:
        try:
            instances_data = json.loads(instances_json)
            for inst in instances_data:
                instances.append(OllamaAccount(**inst))
            logger.info(f"Loaded {len(instances)} instances from OLLAMA_INSTANCES")
            return instances
        except Exception as e:
            logger.error(f"Failed to parse OLLAMA_INSTANCES: {e}")
    
    # Default: use localhost
    instances.append(OllamaAccount(
        name="default",
        base_url="http://localhost:11434",
    ))
    logger.info("Using default Ollama instance at localhost:11434")
    
    return instances


@app.on_event("startup")
async def startup_event():
    """Initialize Ollama manager and database on startup"""
    global ollama_manager
    
    # Initialize database
    init_db()
    
    instances = load_instances_from_file()
    
    if not instances:
        raise RuntimeError("No Ollama instances configured!")
    
    ollama_manager = OllamaManager(instances)
    logger.info(f"Started with {len(instances)} instances: {[i.name for i in instances]}")


@app.get("/")
async def root():
    """Serve dashboard HTML"""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    else:
        return {"message": "Ollama Router API. Use /dashboard for metrics or /health for status."}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not ollama_manager:
        return {"status": "initializing", "timestamp": datetime.now().isoformat()}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "instances": len(ollama_manager.instances),
    }


@app.get("/metrics")
async def get_metrics():
    """Get instance metrics and status"""
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "instances": ollama_manager.get_metrics(),
    }


@app.post("/api/chat")
async def chat(request: dict):
    """
    Anthropic API compatible /api/chat endpoint
    Proxies to Ollama with automatic instance switching
    Supports tool calling (function calling)
    """
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    max_retries = len(ollama_manager.instances)
    
    for attempt in range(max_retries):
        try:
            # Get next available instance
            instance = await ollama_manager.get_next_available_instance()
            logger.info(f"Using instance: {instance.name} (attempt {attempt + 1}/{max_retries})")
            
            # Prepare headers with API key if present
            headers = {}
            if instance.api_key:
                headers["Authorization"] = f"Bearer {instance.api_key}"
            
            # Forward request to Ollama (tools are passed through)
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{instance.base_url}/api/chat",
                    json=request,
                    headers=headers,
                )
            
            # Check for rate limit
            if response.status_code == 429:
                logger.warning(f"Rate limit hit on instance {instance.name}")
                await ollama_manager.mark_rate_limited(instance.name, "rate_limit")
                
                # Try next instance
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Brief backoff
                    continue
                else:
                    return JSONResponse(
                        status_code=429,
                        content={"error": "All instances have reached their rate limits"}
                    )
            
            # Success
            elif response.status_code >= 200 and response.status_code < 300:
                # Parse response to extract token and tool call info
                output_tokens = 0
                tool_calls_count = 0
                try:
                    resp_data = response.json()
                    if "message" in resp_data:
                        msg = resp_data["message"]
                        # Count output tokens
                        if "content" in msg:
                            output_tokens = estimate_tokens(msg["content"] if isinstance(msg["content"], str) else str(msg["content"]))
                        # Count tool calls
                        if "tool_calls" in msg and msg["tool_calls"]:
                            tool_calls_count = len(msg["tool_calls"])
                except:
                    pass
                
                # Estimate input tokens from request
                input_tokens = estimate_tokens(str(request.get("messages", [])))
                
                await ollama_manager.record_success(instance.name, input_tokens, output_tokens, tool_calls_count)
                
                # Stream response if requested
                if request.get("stream", False):
                    return StreamingResponse(
                        _stream_response(response),
                        media_type="application/json",
                        headers={"X-Instance-Used": instance.name}
                    )
                else:
                    return JSONResponse(
                        content=response.json(),
                        headers={"X-Instance-Used": instance.name}
                    )
            
            # Other errors
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                await ollama_manager.mark_rate_limited(instance.name, f"http_{response.status_code}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    return JSONResponse(
                        status_code=response.status_code,
                        content=response.json() if response.text else {"error": "Unknown error"}
                    )
        
        except httpx.TimeoutException:
            logger.warning(f"Timeout on instance {instance.name}, trying next...")
            await ollama_manager.mark_rate_limited(instance.name, "timeout")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                raise HTTPException(status_code=504, detail="All instances timed out")
        
        except Exception as e:
            logger.error(f"Error on instance {instance.name}: {e}")
            await ollama_manager.mark_rate_limited(instance.name, str(e))
            if attempt < max_retries - 1:
                continue
            else:
                raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=503, detail="All retry attempts failed")


@app.post("/v1/messages")
async def messages_v1(request: dict):
    """
    Anthropic API v1 compatible endpoint for Claude Code
    Converts Anthropic format to Ollama format and proxies
    Full tool calling support with multi-turn agent loops
    """
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Check for duplicate requests - prevent sending same request twice
    if request_cache.is_duplicate(request):
        logger.warning("Duplicate request detected - ignoring")
        return JSONResponse(status_code=400, content={"error": "Duplicate request"})
    
    request_cache.add(request)
    
    # Convert Anthropic format to Ollama format
    messages = request.get("messages", [])
    anthropic_tools = request.get("tools", [])
    claude_model = request.get("model", "claude-3-5-sonnet-20241022")
    
    # Map Claude model name to Ollama Cloud model
    ollama_model = map_model_name(claude_model)
    
    # Convert Anthropic tools to Ollama format
    ollama_tools = convert_anthropic_tools_to_ollama(anthropic_tools)
    
    # Convert Anthropic messages to Ollama format with proper tool result handling
    ollama_messages = []
    total_input_tokens = 0
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Handle assistant messages with tool calls
        if role == "assistant" and isinstance(content, list):
            # Check for tool_use items in content
            text_parts = []
            tool_calls = []
            
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "tool_use":
                        # Convert Anthropic tool_use to Ollama tool_calls format
                        tool_calls.append({
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("input", {})
                            }
                        })
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            
            # Create assistant message
            msg_obj = {
                "role": "assistant",
                "content": " ".join(text_parts) if text_parts else ""
            }
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            
            ollama_messages.append(msg_obj)
            total_input_tokens += estimate_tokens(" ".join(text_parts))
        
        # Handle user messages with tool results
        elif role == "user" and isinstance(content, list):
            # Process tool results and user content
            text_parts = []
            tool_results = []
            
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "tool_result":
                        tool_result_content = item.get("content", [])
                        if isinstance(tool_result_content, list):
                            result_text = " ".join([x.get("text", "") if isinstance(x, dict) else str(x) for x in tool_result_content])
                        else:
                            result_text = str(tool_result_content)
                        
                        tool_results.append({
                            "role": "tool",
                            "content": result_text,
                            "tool_name": item.get("tool_use_id", "")
                        })
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            
            # Add user message if there's text
            if text_parts:
                user_content = " ".join(text_parts)
                ollama_messages.append({
                    "role": "user",
                    "content": user_content
                })
                total_input_tokens += estimate_tokens(user_content)
            
            # Add tool result messages
            for tool_result in tool_results:
                ollama_messages.append(tool_result)
                total_input_tokens += estimate_tokens(tool_result["content"])
        
        else:
            # Simple message
            msg_obj = {"role": role}
            
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    else:
                        text_parts.append(str(item))
                content = " ".join(text_parts)
            
            msg_obj["content"] = str(content)
            total_input_tokens += estimate_tokens(str(content))
            ollama_messages.append(msg_obj)
    
    # Create Ollama-compatible request
    ollama_request = {
        "model": ollama_model,
        "messages": ollama_messages,
        "stream": request.get("stream", False)
    }
    
    # Add tools if provided
    if ollama_tools:
        ollama_request["tools"] = ollama_tools
    
    max_retries = len(ollama_manager.instances)
    
    for attempt in range(max_retries):
        try:
            instance = await ollama_manager.get_next_available_instance()
            logger.info(f"V1 Messages using instance: {instance.name}")
            
            headers = {}
            if instance.api_key:
                headers["Authorization"] = f"Bearer {instance.api_key}"
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{instance.base_url}/api/chat",
                    json=ollama_request,
                    headers=headers,
                )
            
            if response.status_code == 429:
                logger.warning(f"Rate limit on instance {instance.name}, switching...")
                await ollama_manager.mark_rate_limited(instance.name, "rate_limit")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    return JSONResponse(status_code=429, content={"error": "Rate limited"})
            
            elif response.status_code >= 200 and response.status_code < 300:
                # Handle streaming vs non-streaming
                if request.get("stream", False):
                    # Return streaming response with SSE media type
                    # Estimate tokens from request for now (will be updated by streaming)
                    total_input_tokens = estimate_tokens(json.dumps(request.get("messages", [])))
                    
                    return StreamingResponse(
                        _convert_streaming_response_with_metrics(
                            response, 
                            anthropic_tools,
                            ollama_manager,
                            instance.name,
                            total_input_tokens
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Instance-Used": instance.name}
                    )
                else:
                    # Parse single response and convert tool calls
                    try:
                        data = response.json()
                        
                        # Estimate output tokens and count tool calls
                        output_tokens = 0
                        tool_calls_count = 0
                        
                        # Convert Ollama response format to Anthropic format
                        if "message" in data:
                            msg = data["message"]
                            
                            # Convert tool_calls from Ollama format to Anthropic format
                            if "tool_calls" in msg and msg["tool_calls"]:
                                anthropic_tool_calls = convert_ollama_tool_calls_to_anthropic(msg["tool_calls"])
                                tool_calls_count = len(anthropic_tool_calls)
                                
                                # Update message with converted tool calls
                                if "content" not in msg:
                                    msg["content"] = []
                                if not isinstance(msg["content"], list):
                                    msg["content"] = []
                                
                                # Add tool_use items to content
                                msg["content"].extend(anthropic_tool_calls)
                                
                                # Remove Ollama-format tool_calls
                                del msg["tool_calls"]
                            
                            # Estimate tokens
                            if "content" in msg:
                                if isinstance(msg["content"], list):
                                    for item in msg["content"]:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            output_tokens += estimate_tokens(item.get("text", ""))
                                else:
                                    output_tokens = estimate_tokens(msg["content"])
                        
                        # Record success metrics
                        await ollama_manager.record_success(instance.name, total_input_tokens, output_tokens, tool_calls_count)
                        
                        # Ensure tools are included in response if provided
                        if anthropic_tools and "tools" not in data:
                            data["tools"] = anthropic_tools
                        
                        return JSONResponse(content=data)
                    except json.JSONDecodeError:
                        # Response might be streaming even though stream=false
                        # Try to read last complete JSON object
                        lines = response.text.strip().split('\n')
                        if lines:
                            for line in reversed(lines):
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        # Convert tool calls
                                        if "message" in data and "tool_calls" in data["message"]:
                                            tool_calls = data["message"]["tool_calls"]
                                            if tool_calls:
                                                data["message"]["content"] = convert_ollama_tool_calls_to_anthropic(tool_calls)
                                                del data["message"]["tool_calls"]
                                        # Add tools
                                        if anthropic_tools and "tools" not in data:
                                            data["tools"] = anthropic_tools
                                        return JSONResponse(content=data)
                                    except json.JSONDecodeError:
                                        continue
                        raise HTTPException(status_code=500, detail="Invalid response from Ollama")
            
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                await ollama_manager.mark_rate_limited(instance.name, f"http_{response.status_code}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    return JSONResponse(
                        status_code=response.status_code,
                        content={"error": f"HTTP {response.status_code}"}
                    )
        
        except Exception as e:
            logger.error(f"Error: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=503, detail="All retry attempts failed")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: dict):
    """
    Anthropic API token counting endpoint (stub)
    Returns estimated token count
    """
    messages = request.get("messages", [])
    
    # Simple estimate: ~4 chars per token
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    estimated_tokens = max(1, total_chars // 4)
    
    return {
        "input_tokens": estimated_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0
    }


@app.post("/api/generate")
async def generate(request: dict):
    """
    Ollama /api/generate endpoint with rate limit handling
    """
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    max_retries = len(ollama_manager.instances)
    
    for attempt in range(max_retries):
        try:
            instance = await ollama_manager.get_next_available_instance()
            logger.info(f"Generate using instance: {instance.name}")
            
            # Prepare headers with API key if present
            headers = {}
            if instance.api_key:
                headers["Authorization"] = f"Bearer {instance.api_key}"
            
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{instance.base_url}/api/generate",
                    json=request,
                    headers=headers,
                )
            
            if response.status_code == 429:
                logger.warning(f"Rate limit on instance {instance.name}")
                await ollama_manager.mark_rate_limited(instance.name, "rate_limit")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    raise HTTPException(status_code=429, detail="All instances rate limited")
            
            elif response.status_code >= 200 and response.status_code < 300:
                # Parse response to extract token info
                output_tokens = 0
                try:
                    resp_data = response.json()
                    if "response" in resp_data:
                        output_tokens = estimate_tokens(resp_data["response"])
                except:
                    pass
                
                # Estimate input tokens from request
                input_tokens = estimate_tokens(request.get("prompt", ""))
                
                await ollama_manager.record_success(instance.name, input_tokens, output_tokens, 0)
                
                if request.get("stream", False):
                    return StreamingResponse(
                        _stream_response(response),
                        media_type="application/json",
                        headers={"X-Instance-Used": instance.name}
                    )
                else:
                    return JSONResponse(
                        content=response.json(),
                        headers={"X-Instance-Used": instance.name}
                    )
            
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                await ollama_manager.mark_rate_limited(instance.name, f"http_{response.status_code}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise HTTPException(status_code=response.status_code, detail="All instances failed")
        
        except Exception as e:
            logger.error(f"Error: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=503, detail="All retries failed")


async def _stream_response(response: httpx.Response):
    """Stream response from Ollama"""
    async for line in response.aiter_lines():
        if line:
            yield line + "\n"


async def _convert_streaming_response_with_metrics(response: httpx.Response, anthropic_tools: List[dict], manager: "OllamaManager", instance_name: str, input_tokens: int):
    """
    Wrapper around _convert_streaming_response that records metrics after streaming completes
    """
    output_tokens_container = {"total": 0, "tool_calls": 0}
    
    async for chunk in _convert_streaming_response(response, anthropic_tools, output_tokens_container):
        yield chunk
    
    # Record metrics after streaming completes with actual token counts
    await manager.record_success(instance_name, input_tokens, output_tokens_container["total"], output_tokens_container["tool_calls"])


async def _convert_streaming_response(response: httpx.Response, anthropic_tools: List[dict] = None, metrics_container: dict = None):
    """
    Convert Ollama streaming response to Anthropic SSE format
    Ollama returns: {"message": {"content": "text"}, "done": false}
    Convert to Anthropic SSE format with full tool calling support
    Handles tool_calls in streaming response and converts to Anthropic format
    metrics_container: Optional dict with "total" and "tool_calls" keys to track tokens
    """
    # Send message_start event with tools if provided
    message_obj = {
        "id": "msg_router",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "minimax-m2.1:cloud",
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }
    
    # Include tools in message if provided
    if anthropic_tools:
        message_obj["tools"] = anthropic_tools
    
    message_start = {
        "type": "message_start",
        "message": message_obj
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
    
    # Send content_block_start for text content
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
    
    output_tokens = 0
    tool_block_index = 1
    tools_emitted = False
    
    async for line in response.aiter_lines():
        if line.strip():
            try:
                data = json.loads(line)
                
                # Extract message content from Ollama format
                if "message" in data:
                    msg = data["message"]
                    
                    # Handle text content
                    if "content" in msg and msg["content"]:
                        content = msg["content"]
                        # Send content_block_delta in Anthropic format
                        delta = {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "text_delta",
                                "text": content
                            }
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                        output_tokens += max(1, len(content) // 4)
                    
                    # Handle tool calls
                    if "tool_calls" in msg and msg["tool_calls"] and not tools_emitted:
                        anthropic_tool_calls = convert_ollama_tool_calls_to_anthropic(msg["tool_calls"])
                        
                        # Update metrics if container provided
                        if metrics_container is not None:
                            metrics_container["tool_calls"] = len(anthropic_tool_calls)
                        
                        for tool_call in anthropic_tool_calls:
                            # Send content_block_start for tool use
                            tool_block_start = {
                                "type": "content_block_start",
                                "index": tool_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_call["id"],
                                    "name": tool_call["name"],
                                    "input": {}
                                }
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(tool_block_start)}\n\n"
                            
                            # Send tool input as delta
                            tool_delta = {
                                "type": "content_block_delta",
                                "index": tool_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(tool_call["input"])
                                }
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(tool_delta)}\n\n"
                            
                            # Send tool block stop
                            tool_block_stop = {
                                "type": "content_block_stop",
                                "index": tool_block_index
                            }
                            yield f"event: content_block_stop\ndata: {json.dumps(tool_block_stop)}\n\n"
                            
                            tool_block_index += 1
                        
                        tools_emitted = True
                
                # Send stop events when done
                if data.get("done", False):
                    # Send text content_block_stop if text was sent
                    content_block_stop = {
                        "type": "content_block_stop",
                        "index": 0
                    }
                    yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"
                    
                    # Update metrics container with final output tokens
                    if metrics_container is not None:
                        metrics_container["total"] = output_tokens
                    
                    # Send message_delta with usage
                    message_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens}
                    }
                    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
                    
                    # Send final message_stop
                    message_stop = {"type": "message_stop"}
                    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse streaming line: {line} - {e}")
                continue


@app.get("/api/tags")
async def list_models():
    """List available models from all instances"""
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    all_models = set()
    
    for instance in ollama_manager.instances:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{instance.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    for model in models:
                        all_models.add(model.get("name", ""))
        except Exception as e:
            logger.warning(f"Could not fetch models from {instance.name}: {e}")
    
    return {
        "models": [{"name": m, "size": 0, "digest": "", "modified_at": ""} for m in sorted(all_models)]
    }


@app.get("/instances")
async def list_instances():
    """List all configured instances"""
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "total": len(ollama_manager.instances),
        "instances": [
            {
                "name": inst.name,
                "base_url": inst.base_url,
                "max_requests_per_minute": inst.max_requests_per_minute,
            }
            for inst in ollama_manager.instances
        ]
    }


@app.get("/instances/{instance_name}/metrics")
async def get_instance_metrics(instance_name: str):
    """Get metrics for specific instance"""
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if instance_name not in ollama_manager.metrics:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    metrics = ollama_manager.metrics[instance_name]
    return {
        "name": metrics.name,
        "requests_made": metrics.requests_made,
        "is_rate_limited": metrics.is_rate_limited,
        "consecutive_errors": metrics.consecutive_errors,
        "last_error": metrics.last_error,
        "last_rate_limit": metrics.last_rate_limit_time.isoformat() if metrics.last_rate_limit_time else None,
    }


@app.get("/dashboard")
async def dashboard():
    """
    Dashboard endpoint showing usage across all accounts
    Returns aggregated metrics and per-account statistics (from memory + database)
    """
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Calculate aggregated stats
    total_requests = 0
    total_tokens_uploaded = 0
    total_tokens_downloaded = 0
    total_tool_calls = 0
    rate_limited_count = 0
    healthy_count = 0
    total_errors = 0
    
    accounts = []
    for instance in ollama_manager.instances:
        metrics = ollama_manager.metrics[instance.name]
        
        # Load persisted data from database
        try:
            db_metrics = TokenMetrics.get(TokenMetrics.account_name == instance.name)
            # Use database values if available and newer (more complete history)
            requests_made = db_metrics.requests_made + (metrics.requests_made if metrics.requests_made > db_metrics.requests_made else 0)
            tokens_uploaded = db_metrics.tokens_uploaded
            tokens_downloaded = db_metrics.tokens_downloaded
            tool_calls = db_metrics.tool_calls
            rate_limited = db_metrics.is_rate_limited
            last_error = db_metrics.last_error
            rate_limit_count_db = db_metrics.rate_limit_count
            created_at = db_metrics.created_at
        except:
            # Fallback to in-memory metrics if no database entry
            requests_made = metrics.requests_made
            tokens_uploaded = metrics.tokens_uploaded
            tokens_downloaded = metrics.tokens_downloaded
            tool_calls = metrics.tool_calls
            rate_limited = metrics.is_rate_limited
            last_error = metrics.last_error
            rate_limit_count_db = metrics.consecutive_errors
            created_at = metrics.created_at
        
        total_requests += requests_made
        total_tokens_uploaded += tokens_uploaded
        total_tokens_downloaded += tokens_downloaded
        total_tool_calls += tool_calls
        total_errors += rate_limit_count_db
        
        if rate_limited:
            rate_limited_count += 1
        else:
            healthy_count += 1
        
        # Estimate usage percentage based on rate limiting status
        if rate_limited:
            usage_percent = 100
        elif rate_limit_count_db > 0:
            usage_percent = 80
        else:
            usage_percent = min(50, (requests_made / 10))  # Rough estimate
        
        accounts.append({
            "name": instance.name,
            "requests_made": requests_made,
            "tokens_uploaded": tokens_uploaded,
            "tokens_downloaded": tokens_downloaded,
            "tool_calls": tool_calls,
            "is_rate_limited": rate_limited,
            "usage_percent": usage_percent,
            "consecutive_errors": rate_limit_count_db,
            "last_error": last_error,
            "last_rate_limit": metrics.last_rate_limit_time.isoformat() if metrics.last_rate_limit_time else None,
            "uptime_seconds": (datetime.now() - created_at).total_seconds(),
        })
    
    # Calculate totals
    total_accounts = len(ollama_manager.instances)
    overall_health = "healthy" if rate_limited_count == 0 else "degraded" if rate_limited_count < total_accounts else "limited"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_accounts": total_accounts,
            "healthy_accounts": healthy_count,
            "rate_limited_accounts": rate_limited_count,
            "overall_health": overall_health,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "tokens_uploaded": total_tokens_uploaded,
            "tokens_downloaded": total_tokens_downloaded,
            "tool_calls": total_tool_calls,
            "estimated_capacity": f"{healthy_count}/{total_accounts} accounts available",
            "rate_limit_per_account": "30 requests/minute",
            "estimated_total_capacity": f"~{healthy_count * 30} requests/minute",
        },
        "accounts": sorted(accounts, key=lambda x: x["usage_percent"], reverse=True),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

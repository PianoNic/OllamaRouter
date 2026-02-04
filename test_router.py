"""
Test suite for Claude API Router
"""

import pytest
import json
from fastapi.testclient import TestClient
from main import app, AccountConfig, AccountManager


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def test_accounts():
    """Create test accounts"""
    return [
        AccountConfig(
            name="test_account_1",
            api_key="sk-ant-test-1",
        ),
        AccountConfig(
            name="test_account_2",
            api_key="sk-ant-test-2",
        ),
    ]


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_list_accounts(client):
    """Test listing accounts"""
    response = client.get("/accounts")
    assert response.status_code in [200, 503]  # May not be initialized


def test_account_manager_initialization(test_accounts):
    """Test AccountManager initialization"""
    manager = AccountManager(test_accounts)
    assert len(manager.accounts) == 2
    assert manager.accounts[0].name == "test_account_1"


@pytest.mark.asyncio
async def test_account_manager_rotation(test_accounts):
    """Test account rotation"""
    manager = AccountManager(test_accounts)
    
    # Get first account
    acc1 = await manager.get_next_available_account()
    assert acc1.name == "test_account_1"
    
    # Mark as rate limited and get next
    await manager.mark_rate_limited(acc1.name)
    acc2 = await manager.get_next_available_account()
    assert acc2.name == "test_account_2"


@pytest.mark.asyncio
async def test_account_metrics_tracking(test_accounts):
    """Test metrics tracking"""
    manager = AccountManager(test_accounts)
    
    # Record success
    await manager.record_success("test_account_1", tokens_used=100)
    metrics = manager.get_metrics()
    
    assert metrics["test_account_1"]["tokens_used"] == 100
    assert metrics["test_account_1"]["requests_made"] == 1


def test_account_config_validation():
    """Test AccountConfig validation"""
    config = AccountConfig(
        name="test",
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
        max_tokens_per_minute=50000,
        requests_per_minute=30,
    )
    
    assert config.name == "test"
    assert config.api_key == "sk-ant-test"
    assert config.max_tokens_per_minute == 50000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

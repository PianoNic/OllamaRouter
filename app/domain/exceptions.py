"""Domain-level exceptions. No framework coupling."""


class DomainError(Exception):
    """Base for all domain errors."""


class NoAccountsConfigured(DomainError):
    """Raised at startup when no Ollama accounts are available."""


class AllInstancesRateLimited(DomainError):
    """Raised when every configured account is currently rate-limited."""


class UpstreamError(DomainError):
    """Wraps a non-recoverable upstream failure."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status

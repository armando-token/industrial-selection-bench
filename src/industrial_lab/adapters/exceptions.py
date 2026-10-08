"""Adapter-related exceptions for Industrial Selection Lab.

Adheres to MEGAPLAN.md §0, §9.5, §20.3, §24.2.
"""


class AdapterError(Exception):
    """Base exception for all adapter errors."""


class ProviderBlockedError(RuntimeError, AdapterError):
    """Raised when an external provider's credentials or access are missing or blocked.

    Adheres to MEGAPLAN.md §0 and §20.3: JEV or LLM cannot be run without credentials,
    and must not be silently mocked or substituted with another model.
    """


class ModelValidationError(ValueError, AdapterError):
    """Raised when a model identifier is invalid (e.g., 'latest' or unversioned)."""


class ResponseValidationError(ValueError, AdapterError):
    """Raised when an API response fails validation checks (missing questions, invalid probabilities, NaN, etc.)."""


class APIRequestError(AdapterError):
    """Raised when an API request fails after retries or returns an unhandled HTTP error."""

    def __init__(self, message: str, status_code: int | None = None, response_text: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class ProviderContractError(APIRequestError):
    """Raised when an API request fails due to 422 contract error. Must not be retried with the same body."""

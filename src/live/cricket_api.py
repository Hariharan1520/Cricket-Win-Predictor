"""
Cricket Data API Client for https://cricketdata.org/

Handles authentication, HTTP requests, error handling, rate limiting, and safe logging.
NEVER exposes or logs the API key in plain text.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Search for .env in current directory and parent directories
_env_path = Path(".env")
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()


class CricketApiError(Exception):
    """Base exception for Cricket Data API errors."""
    pass


class CricketApiAuthError(CricketApiError):
    """Raised when API key is missing or invalid."""
    pass


class CricketApiRateLimitError(CricketApiError):
    """Raised when daily quota or rate limit is reached."""
    pass


def mask_sensitive_url(url: str) -> str:
    """Masks the apikey parameter in a URL string for safe logging and error reporting."""
    if not url:
        return ""
    return re.sub(r"(apikey=)[^&]+", r"\1***REDACTED***", str(url), flags=re.IGNORECASE)


class CricketApiClient:
    """
    Client for interacting with Cricket Data API (v1).
    Documentation: https://cricketdata.org/
    """

    BASE_URL = "https://api.cricapi.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.api_key = api_key or os.getenv("CRICKET_API_KEY")
        if not self.api_key:
            raise CricketApiAuthError(
                "CRICKET_API_KEY environment variable is not set. "
                "Please configure it in your .env file or environment."
            )
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def __repr__(self) -> str:
        key_masked = (
            f"{self.api_key[:4]}...{self.api_key[-4:]}"
            if len(self.api_key) > 8
            else "***"
        )
        return f"<CricketApiClient base_url='{self.base_url}' api_key='{key_masked}'>"

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a GET request against the Cricket Data API.
        Masks API keys in all logs and exception messages.
        """
        if params is None:
            params = {}
        request_params = dict(params)
        request_params["apikey"] = self.api_key

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        masked_url = mask_sensitive_url(
            f"{url}?{urlencode(request_params)}"
        )
        logger.debug(f"Calling Cricket Data API: {masked_url}")

        try:
            response = self.session.get(
                url,
                params=request_params,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"API request timed out: {masked_url}")
            raise CricketApiError(f"API request timed out after {self.timeout}s: {mask_sensitive_url(str(e))}") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"API connection error: {masked_url}")
            raise CricketApiError(f"Failed to connect to Cricket Data API: {mask_sensitive_url(str(e))}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"API request exception: {masked_url}")
            raise CricketApiError(f"API request failed: {mask_sensitive_url(str(e))}") from e

        # Handle HTTP status codes
        if response.status_code == 401 or response.status_code == 403:
            raise CricketApiAuthError(
                f"Authentication failed (HTTP {response.status_code}). Please verify your CRICKET_API_KEY."
            )
        elif response.status_code == 429:
            raise CricketApiRateLimitError(
                f"Rate limit exceeded (HTTP 429). Please check your daily hit quota."
            )
        elif response.status_code != 200:
            raise CricketApiError(
                f"API returned HTTP error {response.status_code}: {response.text[:200]}"
            )

        # Parse JSON
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise CricketApiError(f"Malformed JSON response from API: {response.text[:200]}") from e

        # Check API-level status and rate limits
        api_status = data.get("status")
        if api_status != "success":
            info = data.get("info", {})
            reason = info.get("hitsToday", "") if isinstance(info, dict) else ""
            msg = f"API returned status '{api_status}'"
            if reason:
                msg += f" (Hits today: {reason})"
            if "hitsLimit" in info and info.get("hitsToday", 0) >= info.get("hitsLimit", 100):
                raise CricketApiRateLimitError(f"Cricket Data API daily quota exhausted: {info}")
            raise CricketApiError(msg)

        return data

    def get_current_matches(self, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieves currently live / ongoing matches.
        Endpoint: /currentMatches
        """
        data = self._request("currentMatches", params={"offset": offset})
        return data.get("data", [])

    def get_matches(self, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieves recent and scheduled matches.
        Endpoint: /matches
        """
        data = self._request("matches", params={"offset": offset})
        return data.get("data", [])

    def get_match_info(self, match_id: str) -> Dict[str, Any]:
        """
        Retrieves detailed information and live score for a specific match.
        Endpoint: /match_info?id={match_id}
        """
        if not match_id:
            raise ValueError("match_id cannot be empty")
        data = self._request("match_info", params={"id": match_id})
        return data.get("data", {})

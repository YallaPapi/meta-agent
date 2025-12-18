"""Tests for the license verification module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from metaagent.license import (
    verify_pro_key,
    is_pro,
    get_tier_info,
    check_iteration_limit,
    add_pro_key_hash,
    FREE_TIER_LIMIT,
    PRO_KEY_ENV_VAR,
)


class TestVerifyProKey:
    """Tests for verify_pro_key function."""

    def test_valid_demo_key(self):
        """Test that demo key '1' is valid."""
        assert verify_pro_key("1") is True

    def test_invalid_key(self):
        """Test that invalid key returns False."""
        assert verify_pro_key("invalid-key") is False

    def test_none_key(self):
        """Test that None key returns False."""
        assert verify_pro_key(None) is False

    def test_empty_string_key(self):
        """Test that empty string returns False."""
        assert verify_pro_key("") is False

    def test_reads_from_env_when_none(self):
        """Test that it reads from environment when key is None."""
        with patch.dict(os.environ, {PRO_KEY_ENV_VAR: "1"}):
            assert verify_pro_key(None) is True

        with patch.dict(os.environ, {PRO_KEY_ENV_VAR: "invalid"}):
            assert verify_pro_key(None) is False


class TestIsPro:
    """Tests for is_pro convenience function."""

    def test_is_pro_with_valid_env_key(self):
        """Test is_pro returns True when valid key in env."""
        with patch.dict(os.environ, {PRO_KEY_ENV_VAR: "1"}):
            assert is_pro() is True

    def test_is_pro_with_invalid_env_key(self):
        """Test is_pro returns False when invalid key in env."""
        with patch.dict(os.environ, {PRO_KEY_ENV_VAR: "invalid"}, clear=True):
            assert is_pro() is False

    def test_is_pro_with_no_env_key(self):
        """Test is_pro returns False when no key in env."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear the env var if it exists
            if PRO_KEY_ENV_VAR in os.environ:
                del os.environ[PRO_KEY_ENV_VAR]
            assert is_pro() is False


class TestGetTierInfo:
    """Tests for get_tier_info function."""

    def test_free_tier(self):
        """Test free tier info."""
        info = get_tier_info(None)
        assert info["tier"] == "Free"
        assert info["iteration_limit"] == FREE_TIER_LIMIT
        assert info["is_pro"] is False

    def test_pro_tier(self):
        """Test pro tier info."""
        info = get_tier_info("1")  # Demo key
        assert info["tier"] == "Pro"
        assert info["iteration_limit"] is None
        assert info["is_pro"] is True

    def test_invalid_key_returns_free(self):
        """Test that invalid key returns free tier."""
        info = get_tier_info("invalid-key")
        assert info["tier"] == "Free"
        assert info["is_pro"] is False


class TestCheckIterationLimit:
    """Tests for check_iteration_limit function."""

    def test_pro_tier_allows_any_iteration(self):
        """Test pro tier allows any iteration number."""
        is_allowed, message = check_iteration_limit(100, "1")
        assert is_allowed is True
        assert "Pro tier" in message
        assert "unlimited" in message

    def test_free_tier_allows_up_to_limit(self):
        """Test free tier allows iterations up to limit."""
        for i in range(1, FREE_TIER_LIMIT + 1):
            is_allowed, message = check_iteration_limit(i, None)
            assert is_allowed is True
            assert "Free tier" in message

    def test_free_tier_blocks_over_limit(self):
        """Test free tier blocks iterations over limit."""
        is_allowed, message = check_iteration_limit(FREE_TIER_LIMIT + 1, None)
        assert is_allowed is False
        assert "limit reached" in message
        assert "Upgrade" in message

    def test_shows_remaining_iterations(self):
        """Test message shows remaining iterations."""
        is_allowed, message = check_iteration_limit(3, None)
        remaining = FREE_TIER_LIMIT - 3
        assert str(remaining) in message


class TestAddProKeyHash:
    """Tests for add_pro_key_hash utility function."""

    def test_generates_md5_hash(self):
        """Test that function generates correct MD5 hash."""
        # md5("1") = c4ca4238a0b923820dcc509a6f75849b
        hash_result = add_pro_key_hash("1")
        assert hash_result == "c4ca4238a0b923820dcc509a6f75849b"

    def test_different_keys_different_hashes(self):
        """Test that different keys produce different hashes."""
        hash1 = add_pro_key_hash("key1")
        hash2 = add_pro_key_hash("key2")
        assert hash1 != hash2


class TestFreeTierLimit:
    """Tests for FREE_TIER_LIMIT constant."""

    def test_limit_is_five(self):
        """Test that free tier limit is 5."""
        assert FREE_TIER_LIMIT == 5

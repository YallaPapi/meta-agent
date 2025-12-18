"""License verification for meta-agent freemium model.

This module handles pro license key validation for the meta-agent CLI tool.
Free tier: 5 iterations per PRD
Pro tier: Unlimited iterations

To get a pro key, visit: https://yoursite.gumroad.com/l/meta-agent-pro
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Optional

# Pro key hashes (SHA256 for security)
# Generate with: hashlib.sha256("your-key".encode()).hexdigest()
PRO_KEY_HASHES_SHA256 = {
    # Demo key: "metaagent-pro-demo" -> for testing
    "a1b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9",
    # Add production key hashes here
}

# Legacy MD5 hashes (for backwards compatibility)
PRO_KEY_HASHES_MD5 = {
    "c4ca4238a0b923820dcc509a6f75849b",  # Demo key: "1"
}

# Environment variable name for pro key
PRO_KEY_ENV_VAR = "METAAGENT_PRO_KEY"

# Free tier iteration limit
FREE_TIER_LIMIT = 5

# Purchase URL
PURCHASE_URL = "https://yoursite.gumroad.com/l/meta-agent-pro"

# Cached pro key (to avoid repeated prompts)
_cached_pro_key: Optional[str] = None
_cached_is_pro: Optional[bool] = None


def get_pro_key() -> Optional[str]:
    """Get pro key from environment.

    Returns:
        Pro key if set, None otherwise.
    """
    return os.getenv(PRO_KEY_ENV_VAR)


def prompt_for_key(silent: bool = False) -> Optional[str]:
    """Prompt user for pro key interactively.

    Args:
        silent: If True, don't prompt (return None immediately).

    Returns:
        Pro key entered by user, or None if skipped.
    """
    global _cached_pro_key

    if _cached_pro_key is not None:
        return _cached_pro_key

    if silent or not sys.stdin.isatty():
        return None

    print("\n" + "=" * 60)
    print("META-AGENT LICENSE")
    print("=" * 60)
    print(f"Free tier: {FREE_TIER_LIMIT} iterations per PRD")
    print("Pro tier: Unlimited iterations")
    print(f"\nGet your pro key at: {PURCHASE_URL}")
    print("=" * 60)

    try:
        key = input("\nEnter pro key (or press Enter for free tier): ").strip()
        if key:
            _cached_pro_key = key
            return key
    except (EOFError, KeyboardInterrupt):
        pass

    return None


def verify_pro_key(key: Optional[str] = None, use_sha256: bool = True) -> bool:
    """Verify if a license key is a valid pro key.

    Args:
        key: License key to verify. If None, reads from environment.
        use_sha256: Use SHA256 hashing (more secure). Falls back to MD5.

    Returns:
        True if key is valid pro key.
    """
    if key is None:
        key = get_pro_key()

    if not key:
        return False

    # Try SHA256 first (more secure)
    if use_sha256:
        key_hash_sha256 = hashlib.sha256(key.encode()).hexdigest()
        if key_hash_sha256 in PRO_KEY_HASHES_SHA256:
            return True

    # Fall back to MD5 for backwards compatibility
    key_hash_md5 = hashlib.md5(key.encode()).hexdigest()
    return key_hash_md5 in PRO_KEY_HASHES_MD5


def is_pro(prompt_if_missing: bool = False) -> bool:
    """Check if running in pro mode.

    Args:
        prompt_if_missing: If True, prompt for key if not in environment.

    Returns:
        True if valid pro key is available.
    """
    global _cached_is_pro

    if _cached_is_pro is not None:
        return _cached_is_pro

    # Check environment first
    if verify_pro_key():
        _cached_is_pro = True
        return True

    # Optionally prompt for key
    if prompt_if_missing:
        key = prompt_for_key()
        if key and verify_pro_key(key):
            _cached_is_pro = True
            return True

    _cached_is_pro = False
    return False


def get_tier_info(pro_key: Optional[str] = None) -> dict:
    """Get tier information including limits.

    Args:
        pro_key: Optional pro key to check.

    Returns:
        Dict with tier, iteration_limit, and is_pro.
    """
    is_pro_user = verify_pro_key(pro_key)

    return {
        "tier": "Pro" if is_pro_user else "Free",
        "iteration_limit": None if is_pro_user else FREE_TIER_LIMIT,
        "is_pro": is_pro_user,
    }


def check_iteration_limit(iteration: int, pro_key: Optional[str] = None) -> tuple[bool, str]:
    """Check if iteration is within tier limits.

    Args:
        iteration: Current iteration number (1-indexed).
        pro_key: Optional pro key.

    Returns:
        Tuple of (is_allowed, message).
    """
    tier_info = get_tier_info(pro_key)

    if tier_info["is_pro"]:
        return True, f"Pro tier: Iteration {iteration} (unlimited)"

    if iteration <= FREE_TIER_LIMIT:
        remaining = FREE_TIER_LIMIT - iteration
        return True, f"Free tier: Iteration {iteration}/{FREE_TIER_LIMIT} ({remaining} remaining)"

    return False, (
        f"\nFree tier limit reached ({FREE_TIER_LIMIT} iterations).\n"
        f"Upgrade to Pro for unlimited iterations.\n"
        f"Get your pro key at: {PURCHASE_URL}\n"
    )


def generate_key_hash(key: str, algorithm: str = "sha256") -> str:
    """Generate hash for a new pro key.

    Utility function for adding new keys.

    Args:
        key: The pro key to hash.
        algorithm: Hash algorithm ('sha256' or 'md5').

    Returns:
        Hash of the key.
    """
    if algorithm == "sha256":
        return hashlib.sha256(key.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(key.encode()).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# Backwards compatibility alias
def add_pro_key_hash(key: str) -> str:
    """Generate MD5 hash for a new pro key (legacy).

    Args:
        key: The pro key to hash.

    Returns:
        MD5 hash of the key.
    """
    return generate_key_hash(key, "md5")


def reset_cache() -> None:
    """Reset cached license state.

    Useful for testing or when user wants to re-enter key.
    """
    global _cached_pro_key, _cached_is_pro
    _cached_pro_key = None
    _cached_is_pro = None


def display_license_status(pro_key: Optional[str] = None) -> None:
    """Display current license status to console.

    Args:
        pro_key: Optional pro key to check.
    """
    tier_info = get_tier_info(pro_key)

    print("\n" + "-" * 40)
    print(f"License: {tier_info['tier']}")
    if tier_info['is_pro']:
        print("Iterations: Unlimited")
    else:
        print(f"Iterations: {FREE_TIER_LIMIT} per PRD")
        print(f"Upgrade at: {PURCHASE_URL}")
    print("-" * 40 + "\n")

"""License verification for meta-agent freemium model.

This module handles pro license key validation for the meta-agent CLI tool.
Free tier: 5 iterations per PRD
Pro tier: Unlimited iterations

To get a pro key, visit: https://yoursite.gumroad.com/l/meta-agent-pro
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

# Pro key hashes - add validated key hashes here
# Generate with: hashlib.md5("your-key".encode()).hexdigest()
PRO_KEY_HASHES = {
    "c4ca4238a0b923820dcc509a6f75849b",  # Demo key: "1"
    # Add production key hashes here
}

# Environment variable name for pro key
PRO_KEY_ENV_VAR = "METAAGENT_PRO_KEY"

# Free tier iteration limit
FREE_TIER_LIMIT = 5


def get_pro_key() -> Optional[str]:
    """Get pro key from environment.

    Returns:
        Pro key if set, None otherwise.
    """
    return os.getenv(PRO_KEY_ENV_VAR)


def verify_pro_key(key: Optional[str] = None) -> bool:
    """Verify if a license key is a valid pro key.

    Args:
        key: License key to verify. If None, reads from environment.

    Returns:
        True if key is valid pro key.
    """
    if key is None:
        key = get_pro_key()

    if not key:
        return False

    key_hash = hashlib.md5(key.encode()).hexdigest()
    return key_hash in PRO_KEY_HASHES


def is_pro() -> bool:
    """Check if running in pro mode.

    Convenience function that checks environment variable.

    Returns:
        True if valid pro key is set.
    """
    return verify_pro_key()


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
        f"Free tier limit reached ({FREE_TIER_LIMIT} iterations). "
        "Upgrade to Pro for unlimited iterations.\n"
        "Get your pro key at: https://yoursite.gumroad.com/l/meta-agent-pro"
    )


def add_pro_key_hash(key: str) -> str:
    """Generate hash for a new pro key.

    Utility function for adding new keys.

    Args:
        key: The pro key to hash.

    Returns:
        MD5 hash of the key.
    """
    return hashlib.md5(key.encode()).hexdigest()

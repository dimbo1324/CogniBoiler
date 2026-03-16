"""
Password hashing and verification using Argon2 via pwdlib.

Why Argon2 over bcrypt:
  - Winner of the Password Hashing Competition (2015)
  - Memory-hard: tunable memory cost makes GPU brute-force expensive
  - Three attack resistance modes; we use Argon2id (hybrid of i and d)

Usage:
    hashed = hash_password("secret123")
    is_valid = verify_password("secret123", hashed)   # True
    is_valid = verify_password("wrong",     hashed)   # False
"""

from __future__ import annotations

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

# ─── Hasher instance ─────────────────────────────────────────────────────────
# Argon2id is the recommended variant:
#   - Argon2i  → side-channel attack resistant (password hashing)
#   - Argon2d  → GPU brute-force resistant
#   - Argon2id → hybrid; resistant to both
#
# These parameters follow OWASP recommendations for interactive logins:
#   time_cost=2, memory_cost=65536 (64 MB), parallelism=2
_hasher = PasswordHash([Argon2Hasher()])


def hash_password(plain: str) -> str:
    """
    Hash a plain-text password using Argon2id.

    The returned string is a self-contained encoded hash that includes
    the algorithm identifier, parameters, random salt, and hash digest.
    No separate salt storage is needed.

    Args:
        plain: Plain-text password from the user.

    Returns:
        Argon2id encoded hash string, safe to store in the database.
    """
    return _hasher.hash(plain)  # type: ignore[no-any-return]


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plain-text password against an Argon2id hash.

    Performs a constant-time comparison to prevent timing attacks.
    Never raises — returns False on any mismatch or encoding error.

    Args:
        plain:  Plain-text password to check (from login form).
        hashed: Stored Argon2id hash from the database.

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        return _hasher.verify(plain, hashed)  # type: ignore[no-any-return]
    except Exception:
        # Malformed hash or unsupported algorithm — treat as mismatch
        return False

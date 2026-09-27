"""
Password hashing and verification with Argon2id (pwdlib).

Argon2id is memory-hard, which makes GPU guessing expensive, and resists both
side-channel (Argon2i) and GPU (Argon2d) attacks.

Usage:
    hashed = hash_password("secret123")
    is_valid = verify_password("secret123", hashed)   # True
    is_valid = verify_password("wrong",     hashed)   # False
"""

from __future__ import annotations

import logging

from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from pwdlib.hashers.argon2 import Argon2Hasher

logger = logging.getLogger(__name__)

# Pinned, not left to the library's defaults: sign-in verifies an unknown user against a
# dummy hash made with these values, and a silent change of cost after an upgrade would
# let response time tell unknown users from wrong passwords. 64 MiB, 3 passes, 4 lanes.
TIME_COST = 3
MEMORY_COST_KIB = 65536
PARALLELISM = 4

_hasher = PasswordHash(
    [
        Argon2Hasher(
            time_cost=TIME_COST, memory_cost=MEMORY_COST_KIB, parallelism=PARALLELISM
        )
    ]
)


def hash_password(plain: str) -> str:
    """
    Hash a plain-text password using Argon2id.

    The returned string holds the algorithm, its parameters, a random salt and the
    digest, so no separate salt storage is needed.
    """
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Whether a plain-text password matches a stored Argon2 hash.

    A wrong password, or a stored value that is not a hash this module can read, is a
    mismatch; the latter is also logged, without the value. Any other failure (memory
    exhaustion, for one) propagates instead of passing for a wrong password.
    """
    try:
        return _hasher.verify(plain, hashed)
    except UnknownHashError:
        logger.warning("A stored password hash is malformed or of an unknown scheme")
        return False

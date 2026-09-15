"""Gateway test fixtures that must not depend on a developer's .env."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from api_gateway.config import settings
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def _pem_key_pair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private.decode("ascii"), public.decode("ascii")


@pytest.fixture(scope="session", autouse=True)
def ephemeral_jwt_keys() -> Iterator[None]:
    """Sign and verify tokens with a key pair that exists only for this test run."""
    original = (settings.jwt_private_key, settings.jwt_public_key)
    settings.jwt_private_key, settings.jwt_public_key = _pem_key_pair()
    yield
    settings.jwt_private_key, settings.jwt_public_key = original

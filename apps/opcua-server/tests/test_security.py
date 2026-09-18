"""The OPC UA server's certificate and its refusal of passwords sent in clear."""

from __future__ import annotations

import pytest
from asyncua import ua
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from opcua_server.identity import sends_password_in_clear
from opcua_server.security import (
    APPLICATION_URI,
    SECURITY_POLICIES,
    certificate_from_environment,
    self_signed_certificate,
)


def test_a_password_in_clear_on_an_open_channel_is_detected() -> None:
    token = ua.UserNameIdentityToken(UserName="operator", Password=b"plain")
    assert sends_password_in_clear(token, None)


def test_an_encrypted_password_or_an_encrypted_channel_is_accepted() -> None:
    encrypted = ua.UserNameIdentityToken(
        UserName="operator",
        Password=b"ciphertext",
        EncryptionAlgorithm="http://www.w3.org/2001/04/xmlenc#rsa-oaep",
    )
    assert not sends_password_in_clear(encrypted, None)
    in_clear = ua.UserNameIdentityToken(UserName="operator", Password=b"plain")
    assert not sends_password_in_clear(in_clear, b"client certificate")
    assert not sends_password_in_clear(ua.AnonymousIdentityToken(), None)


def test_only_the_open_and_the_encrypted_endpoints_are_offered() -> None:
    assert SECURITY_POLICIES == [
        ua.SecurityPolicyType.NoSecurity,
        ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
    ]


def test_the_self_signed_certificate_names_the_application() -> None:
    certificate = self_signed_certificate()
    parsed = x509.load_pem_x509_certificate(certificate.certificate_pem)
    names = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    assert APPLICATION_URI in names.value.get_values_for_type(
        x509.UniformResourceIdentifier
    )
    usage = parsed.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.key_encipherment and usage.data_encipherment
    key = load_pem_private_key(certificate.private_key_pem, password=None)
    assert isinstance(key, rsa.RSAPrivateKey) and key.key_size == 2048


def test_the_certificate_comes_from_the_environment_only_as_a_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPCUA_SERVER_CERT", "-----BEGIN CERTIFICATE-----")
    monkeypatch.delenv("OPCUA_SERVER_KEY", raising=False)
    assert certificate_from_environment() is None
    monkeypatch.setenv("OPCUA_SERVER_KEY", "-----BEGIN PRIVATE KEY-----")
    found = certificate_from_environment()
    assert found is not None
    assert found.private_key_pem.startswith(b"-----BEGIN PRIVATE KEY")

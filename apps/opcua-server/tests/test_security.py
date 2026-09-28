"""The OPC UA server's certificate and its refusal of passwords sent in clear."""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from asyncua import ua
from asyncua.common.utils import ServiceError
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from opcua_server.identity import (
    _UserAwareSession,
    sends_password_in_clear,
    uses_refused_password_encryption,
)
from opcua_server.security import (
    APPLICATION_URI,
    SECURITY_POLICIES,
    ServerCertificate,
    certificate_from_environment,
    secure,
    self_signed_certificate,
)
from opcua_server.ua_types import StatusCodes


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


def test_a_session_refuses_a_password_in_clear() -> None:
    params = ua.ActivateSessionParameters()
    params.UserIdentityToken = ua.UserNameIdentityToken(
        UserName="operator1", Password=b"pw", EncryptionAlgorithm=None
    )
    session = object.__new__(_UserAwareSession)
    with pytest.raises(ServiceError) as refused:
        session.activate_session(params, None)
    assert refused.value.code == StatusCodes.BadIdentityTokenRejected


def test_only_oaep_password_encryption_is_accepted() -> None:
    def token(algorithm: str | None) -> ua.UserNameIdentityToken:
        return ua.UserNameIdentityToken(
            UserName="operator", Password=b"x", EncryptionAlgorithm=algorithm
        )

    assert uses_refused_password_encryption(
        token("http://www.w3.org/2001/04/xmlenc#rsa-1_5")
    )
    assert not uses_refused_password_encryption(
        token("http://www.w3.org/2001/04/xmlenc#rsa-oaep")
    )
    assert not uses_refused_password_encryption(
        token("http://opcfoundation.org/UA/security/rsa-oaep-sha2-256")
    )
    assert not uses_refused_password_encryption(token(None))
    assert not uses_refused_password_encryption(ua.AnonymousIdentityToken())


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


def issued(
    certificate: ServerCertificate, not_before: dt.datetime, not_after: dt.datetime
) -> ServerCertificate:
    """The same key pair, re-issued with another validity window."""
    key = load_pem_private_key(certificate.private_key_pem, password=None)
    assert isinstance(key, rsa.RSAPrivateKey)
    original = x509.load_pem_x509_certificate(certificate.certificate_pem)
    builder = (
        x509.CertificateBuilder()
        .subject_name(original.subject)
        .issuer_name(original.issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
    )
    rebuilt = builder.sign(key, hashes.SHA256())
    return ServerCertificate(
        rebuilt.public_bytes(serialization.Encoding.PEM), certificate.private_key_pem
    )


class RecordingServer:
    def __init__(self) -> None:
        self.loaded: list[bytes] = []

    async def set_application_uri(self, uri: str) -> None:
        self.uri = uri

    def set_security_policy(self, policies: object) -> None:
        self.policies = policies

    async def load_certificate(self, pem: bytes, format: str) -> None:
        self.loaded.append(pem)

    async def load_private_key(self, pem: bytes, format: str) -> None:
        self.loaded.append(pem)


class TestCertificateAtStartup:
    async def test_a_matching_valid_pair_is_loaded_without_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pair = self_signed_certificate()
        server = RecordingServer()
        with caplog.at_level(logging.WARNING, logger="opcua_server.security"):
            await secure(server, pair)  # type: ignore[arg-type]
        assert server.loaded == [pair.certificate_pem, pair.private_key_pem]
        assert caplog.text == ""

    async def test_without_a_pair_a_temporary_one_is_made_and_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        server = RecordingServer()
        with caplog.at_level(logging.WARNING, logger="opcua_server.security"):
            await secure(server, None)  # type: ignore[arg-type]
        assert len(server.loaded) == 2
        assert "temporary self-signed certificate" in caplog.text

    async def test_a_key_of_another_certificate_is_refused(self) -> None:
        mixed = ServerCertificate(
            self_signed_certificate().certificate_pem,
            self_signed_certificate().private_key_pem,
        )
        server = RecordingServer()
        with pytest.raises(ValueError, match="does not match") as refused:
            await secure(server, mixed)  # type: ignore[arg-type]
        assert "PRIVATE KEY" not in str(refused.value)
        assert server.loaded == []

    async def test_an_expired_certificate_is_refused(self) -> None:
        now = dt.datetime.now(dt.UTC)
        expired = issued(
            self_signed_certificate(),
            now - dt.timedelta(days=400),
            now - dt.timedelta(days=1),
        )
        with pytest.raises(ValueError, match="expired"):
            await secure(RecordingServer(), expired)  # type: ignore[arg-type]

    async def test_a_certificate_close_to_expiry_is_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        now = dt.datetime.now(dt.UTC)
        expiring = issued(
            self_signed_certificate(),
            now - dt.timedelta(days=300),
            now + dt.timedelta(days=5),
        )
        server = RecordingServer()
        with caplog.at_level(logging.WARNING, logger="opcua_server.security"):
            await secure(server, expiring)  # type: ignore[arg-type]
        assert len(server.loaded) == 2
        assert "expires" in caplog.text

    async def test_a_pem_that_does_not_parse_is_refused(self) -> None:
        broken = ServerCertificate(b"not a certificate", b"not a key")
        with pytest.raises(ValueError, match="OPCUA_SERVER_CERT"):
            await secure(RecordingServer(), broken)  # type: ignore[arg-type]

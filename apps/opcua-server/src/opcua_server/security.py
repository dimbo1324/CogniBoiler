"""
The server's security: an application certificate and two endpoints.

- `None`: anonymous reading, as before. A username may sign in here too, but its password
  travels encrypted with the server's public key (the user token policy names
  Basic256Sha256); a password sent in clear on this endpoint is refused, and so is one
  encrypted with anything but RSA-OAEP.
- `Basic256Sha256 / SignAndEncrypt`: the whole channel is encrypted.

Sign-only is not offered, so a channel with a client certificate is always encrypted,
which is what the refusal of clear passwords relies on. Client certificates are accepted
without a trust list: users are authenticated by their password at the gateway, and the
certificate here protects the password and the traffic, not the client's identity.

The certificate and key come from OPCUA_SERVER_CERT and OPCUA_SERVER_KEY (PEM, written to
.env by dev-secrets). Without them the server makes a temporary self-signed certificate,
which clients will see change at every restart.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import socket
from dataclasses import dataclass
from ipaddress import IPv4Address

from asyncua import ua
from asyncua.server.server import Server
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger(__name__)

APPLICATION_URI = "urn:cogniboiler:opcua-server"
SECURITY_POLICIES = [
    ua.SecurityPolicyType.NoSecurity,
    ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
]
CERT_ENV = "OPCUA_SERVER_CERT"
KEY_ENV = "OPCUA_SERVER_KEY"
EXPIRY_WARNING = dt.timedelta(days=30)


@dataclass(frozen=True)
class ServerCertificate:
    certificate_pem: bytes
    private_key_pem: bytes


def certificate_from_environment() -> ServerCertificate | None:
    certificate = os.environ.get(CERT_ENV, "").strip()
    key = os.environ.get(KEY_ENV, "").strip()
    if not certificate or not key:
        return None
    return ServerCertificate(certificate.encode("ascii"), key.encode("ascii"))


def self_signed_certificate(
    application_uri: str = APPLICATION_URI, days: int = 365
) -> ServerCertificate:
    """An application instance certificate for this host, signed by itself."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "CogniBoiler OPC UA server"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CogniBoiler"),
        ]
    )
    now = dt.datetime.now(dt.UTC)
    names: list[x509.GeneralName] = [
        x509.UniformResourceIdentifier(application_uri),
        x509.DNSName("localhost"),
        x509.DNSName(socket.gethostname()),
        x509.IPAddress(IPv4Address("127.0.0.1")),
    ]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=True,
                data_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return ServerCertificate(
        certificate.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


def check_certificate(
    certificate: ServerCertificate, now: dt.datetime | None = None
) -> None:
    """
    Refuse a pair that cannot work, before any client finds out one handshake at a
    time: a key that is not the certificate's, or a certificate already expired. Warn
    when it expires within EXPIRY_WARNING. Messages never carry key material.
    """
    try:
        parsed = x509.load_pem_x509_certificate(certificate.certificate_pem)
        key = serialization.load_pem_private_key(
            certificate.private_key_pem, password=None
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{CERT_ENV} or {KEY_ENV} is not a PEM certificate and unencrypted key: "
            f"{type(exc).__name__}"
        ) from None
    public = serialization.PublicFormat.SubjectPublicKeyInfo
    if key.public_key().public_bytes(
        serialization.Encoding.DER, public
    ) != parsed.public_key().public_bytes(serialization.Encoding.DER, public):
        raise ValueError(f"the key in {KEY_ENV} does not match {CERT_ENV}")
    moment = now or dt.datetime.now(dt.UTC)
    expires = parsed.not_valid_after_utc
    if expires <= moment:
        raise ValueError(
            f"the certificate in {CERT_ENV} expired at {expires:%Y-%m-%dT%H:%M:%SZ}"
        )
    if expires - moment <= EXPIRY_WARNING:
        logger.warning(
            "The OPC UA server certificate expires at %s UTC; renew it with dev-secrets",
            f"{expires:%Y-%m-%dT%H:%M:%S}",
        )


async def secure(server: Server, certificate: ServerCertificate | None) -> None:
    """Install the certificate and the endpoints; call after init(), before start()."""
    if certificate is None:
        logger.warning(
            "%s and %s are not set: using a temporary self-signed certificate",
            CERT_ENV,
            KEY_ENV,
        )
        certificate = self_signed_certificate()
    check_certificate(certificate)
    await server.set_application_uri(APPLICATION_URI)
    server.set_security_policy(SECURITY_POLICIES)
    await server.load_certificate(certificate.certificate_pem, format="pem")
    await server.load_private_key(certificate.private_key_pem, format="pem")

"""The certificates dev-secrets writes for HTTPS and OPC UA.

The certificate cases need the cryptography package and are skipped without it; run them
with:  uv run python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import importlib.util
import unittest

from scripts._toolkit.envfile import EnvFileError
from scripts.dev_secrets.__main__ import build_generators

HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None
PROFILES = {
    "opcua": {
        "common_name": "CogniBoiler OPC UA server",
        "uris": ["urn:cogniboiler:opcua-server"],
        "dns": ["localhost"],
        "ips": ["127.0.0.1"],
        "key": "rsa-2048",
        "days": 30,
    },
    "web_tls": {
        "common_name": "localhost",
        "dns": ["localhost"],
        "key": "ec-p256",
        "days": 30,
    },
}


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "cryptography is not installed")
class CertificateTest(unittest.TestCase):
    def test_a_certificate_and_its_key_come_from_one_generation(self) -> None:
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
            load_pem_private_key,
        )

        generators = build_generators(
            {"CERT": "cert:opcua", "KEY": "key:opcua"}, PROFILES
        )
        certificate = x509.load_pem_x509_certificate(generators["CERT"]().encode())
        key = load_pem_private_key(generators["KEY"]().encode(), password=None)
        spki = (Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        self.assertEqual(
            certificate.public_key().public_bytes(*spki),
            key.public_key().public_bytes(*spki),
        )
        names = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        )
        self.assertIn(
            "urn:cogniboiler:opcua-server",
            names.value.get_values_for_type(x509.UniformResourceIdentifier),
        )

    def test_the_web_certificate_uses_an_elliptic_curve_key(self) -> None:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        key_pem = build_generators({"KEY": "key:web_tls"}, PROFILES)["KEY"]()
        self.assertIsInstance(
            load_pem_private_key(key_pem.encode(), password=None),
            ec.EllipticCurvePrivateKey,
        )


class UnknownKindTest(unittest.TestCase):
    def test_a_certificate_of_an_unknown_profile_is_refused(self) -> None:
        with self.assertRaises(EnvFileError):
            build_generators({"CERT": "cert:missing"}, PROFILES)


if __name__ == "__main__":
    unittest.main()

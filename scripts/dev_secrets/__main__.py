"""Create or complete the local .env: development secrets and the gateway's JWT keys.

Values already in .env are never overwritten. Only key names are printed, never values,
so the output is safe to paste anywhere.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, info, ok, warn
from scripts._toolkit.envfile import EnvFileError, merge, parse, values
from scripts._toolkit.processes import capture
from scripts._toolkit.reexec import has_module, reexec_under_uv

SCRIPT_DIR = Path(__file__).resolve().parent


def _rsa_pair() -> tuple[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

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
    return private.decode("ascii").strip(), public.decode("ascii").strip()


def self_signed(profile: dict[str, Any]) -> tuple[str, str]:
    """A self-signed certificate and its key, PEM, for local development only."""
    import datetime as dt
    from ipaddress import ip_address

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey
    if profile["key"] == "rsa-2048":
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    else:
        key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, profile["common_name"])])
    names: list[x509.GeneralName] = [
        *(x509.UniformResourceIdentifier(uri) for uri in profile.get("uris", [])),
        *(x509.DNSName(dns) for dns in profile.get("dns", [])),
        *(x509.IPAddress(ip_address(ip)) for ip in profile.get("ips", [])),
    ]
    encrypts = profile["key"] == "rsa-2048"
    now = dt.datetime.now(dt.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=int(profile["days"])))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=encrypts,
                key_encipherment=encrypts,
                data_encipherment=encrypts,
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
    certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return certificate_pem.decode("ascii").strip(), key_pem.decode("ascii").strip()


def build_generators(
    kinds: dict[str, str], certificates: dict[str, dict[str, Any]] | None = None
) -> dict[str, Callable[[], str]]:
    pair: list[tuple[str, str]] = []
    issued: dict[str, tuple[str, str]] = {}
    profiles = certificates or {}

    def rsa_part(index: int) -> str:
        if not pair:
            pair.append(_rsa_pair())
        return pair[0][index]

    def certificate_part(profile: str, index: int) -> str:
        if profile not in issued:
            issued[profile] = self_signed(profiles[profile])
        return issued[profile][index]

    makers: dict[str, Callable[[], str]] = {
        "password": lambda: secrets.token_urlsafe(24),
        "token": lambda: secrets.token_urlsafe(48),
        "rsa_private_pem": lambda: rsa_part(0),
        "rsa_public_pem": lambda: rsa_part(1),
    }
    for profile in profiles:
        makers[f"cert:{profile}"] = lambda p=profile: certificate_part(p, 0)
        makers[f"key:{profile}"] = lambda p=profile: certificate_part(p, 1)
    unknown = sorted(set(kinds.values()) - makers.keys())
    if unknown:
        raise EnvFileError(f"unknown generator kind(s) in secrets.json: {unknown}")
    return {key: makers[kind] for key, kind in kinds.items()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="dev-secrets",
        description="Create or complete .env from .env.example without overwriting values.",
    )
    parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "secrets.json")

    if not has_module("cryptography"):
        code = reexec_under_uv("scripts.dev_secrets", argv, root)
        if code is not None:
            return code
        fail("the cryptography package is unavailable — run: uv sync --all-packages")
        return 1

    heading("dev-secrets")
    template_path = root / str(config["template"])
    target_path = root / str(config["target"])
    if not template_path.exists():
        fail(f"missing {config['template']}")
        return 1

    try:
        template = parse(template_path.read_text(encoding="utf-8"))
        existing = (
            values(parse(target_path.read_text(encoding="utf-8")))
            if target_path.exists()
            else {}
        )
        for group in config["set_together"]:
            present = [key for key in group if existing.get(key)]
            if present and len(present) != len(group):
                fail(
                    f"{', '.join(group)} must be set together, but only "
                    f"{', '.join(present)} is set. Remove it from {config['target']} "
                    "to generate a matching set."
                )
                return 1
        rendered, generated, kept = merge(
            template,
            existing,
            build_generators(config["generators"], config.get("certificates")),
        )
    except EnvFileError as error:
        fail(str(error))
        return 1

    temporary = target_path.with_name(target_path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(target_path)

    for key in generated:
        ok(f"generated {key}")
    for key in kept:
        info(f"kept existing {key}")

    code, _ = capture(["git", "check-ignore", "--quiet", str(config["target"])], root)
    if code != 0:
        warn(f"{config['target']} is NOT ignored by git — never commit it")
    ok(f"{config['target']} written: {len(generated)} generated, {len(kept)} kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

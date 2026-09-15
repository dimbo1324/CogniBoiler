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


def build_generators(kinds: dict[str, str]) -> dict[str, Callable[[], str]]:
    pair: list[tuple[str, str]] = []

    def rsa_part(index: int) -> str:
        if not pair:
            pair.append(_rsa_pair())
        return pair[0][index]

    makers: dict[str, Callable[[], str]] = {
        "password": lambda: secrets.token_urlsafe(24),
        "token": lambda: secrets.token_urlsafe(48),
        "rsa_private_pem": lambda: rsa_part(0),
        "rsa_public_pem": lambda: rsa_part(1),
    }
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
            template, existing, build_generators(config["generators"])
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

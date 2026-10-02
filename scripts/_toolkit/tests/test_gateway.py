"""The gateway client `smoke` and `demo` share: requests, replies, sign-in and .env.

The HTTP cases run against a throwaway server on 127.0.0.1, port 0.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts._toolkit.envfile import EnvFileError
from scripts._toolkit.gateway import (
    GatewayUnreachableError,
    Reply,
    call,
    dig,
    load_env,
    sign_in,
)


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict[str, object]] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def _answer(self, status: int, body: object) -> None:
        raw = json.dumps(body).encode("utf-8") if body is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802 — the stdlib's naming
        _Handler.seen.append(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        )
        if self.path == "/health":
            self._answer(200, {"status": "running"})
        elif self.path == "/empty":
            self._answer(204, None)
        elif self.path == "/text-error":
            raw = b"<html>bad gateway</html>"
            self.send_response(502)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self._answer(404, {"detail": "Not Found", "code": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's naming
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"null")
        _Handler.seen.append(
            {
                "path": self.path,
                "payload": payload,
                "type": self.headers["Content-Type"],
            }
        )
        if self.path == "/auth/login" and payload == {
            "username": "operator",
            "password": "right",
        }:
            self._answer(200, {"access_token": "t-1", "token_type": "bearer"})
        else:
            self._answer(401, {"detail": "Invalid credentials", "code": "auth.failed"})


@contextmanager
def gateway() -> Iterator[str]:
    _Handler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class DigTest(unittest.TestCase):
    def test_nested_keys_are_followed(self) -> None:
        self.assertEqual(
            dig({"boiler": {"pressure_pa": 1.5}}, "boiler", "pressure_pa"), 1.5
        )

    def test_a_missing_key_or_a_non_object_yields_none(self) -> None:
        self.assertIsNone(dig({"boiler": {}}, "boiler", "pressure_pa"))
        self.assertIsNone(dig(None, "status"))
        self.assertIsNone(dig([1, 2], "status"))


class ReplyTest(unittest.TestCase):
    def test_only_an_accepted_answer_counts_as_accepted(self) -> None:
        self.assertTrue(Reply(200, {"accepted": True}).accepted)
        self.assertFalse(Reply(200, {"accepted": False, "reason": "E-Stop"}).accepted)
        self.assertFalse(Reply(503, {"detail": "PLCService is unavailable."}).accepted)

    def test_a_refusal_carries_its_reason(self) -> None:
        self.assertEqual(
            Reply(200, {"accepted": False, "reason": "E-Stop active"}).refusal,
            "HTTP 200: E-Stop active",
        )
        self.assertEqual(
            Reply(503, {"detail": "PLCService is unavailable."}).refusal,
            "HTTP 503: PLCService is unavailable.",
        )
        self.assertEqual(Reply(500, None).refusal, "HTTP 500")


class CallTest(unittest.TestCase):
    def test_a_json_answer_is_decoded(self) -> None:
        with gateway() as base_url:
            reply = call(base_url, "GET", "/health", token="t-1")
        self.assertEqual(reply, Reply(200, {"status": "running"}))
        self.assertEqual(_Handler.seen[0]["authorization"], "Bearer t-1")

    def test_an_empty_answer_has_no_body(self) -> None:
        with gateway() as base_url:
            self.assertEqual(call(base_url, "GET", "/empty"), Reply(204, None))

    def test_an_error_keeps_its_problem_details(self) -> None:
        # smoke's copy dropped the body of an HTTP error, so a refusal could not say
        # why; the shared client keeps it for both scripts.
        with gateway() as base_url:
            reply = call(base_url, "GET", "/missing")
        self.assertEqual(reply.status, 404)
        self.assertEqual(dig(reply.body, "code"), "not_found")
        self.assertEqual(reply.refusal, "HTTP 404: Not Found")

    def test_an_error_that_is_not_json_still_reports_its_status(self) -> None:
        with gateway() as base_url:
            self.assertEqual(call(base_url, "GET", "/text-error"), Reply(502, None))

    def test_a_payload_is_sent_as_json(self) -> None:
        with gateway() as base_url:
            call(base_url, "POST", "/anything", payload={"a": 1})
        self.assertEqual(_Handler.seen[0]["payload"], {"a": 1})
        self.assertEqual(_Handler.seen[0]["type"], "application/json")

    def test_no_answer_at_all_is_unreachable(self) -> None:
        with gateway() as base_url:
            pass
        with self.assertRaises(GatewayUnreachableError):
            call(base_url, "GET", "/health", timeout=2.0)


class SignInTest(unittest.TestCase):
    def test_the_right_password_yields_a_token(self) -> None:
        with gateway() as base_url:
            token, reply = sign_in(base_url, "operator", "right")
        self.assertEqual(token, "t-1")
        self.assertEqual(reply.status, 200)

    def test_a_wrong_password_yields_no_token_and_the_refusal(self) -> None:
        with gateway() as base_url:
            token, reply = sign_in(base_url, "operator", "wrong")
        self.assertIsNone(token)
        self.assertEqual(reply.refusal, "HTTP 401: Invalid credentials")


class LoadEnvTest(unittest.TestCase):
    def test_values_are_read_from_the_named_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("A=1\nB=two\n", encoding="utf-8")
            self.assertEqual(load_env(root, ".env"), {"A": "1", "B": "two"})

    def test_a_missing_file_is_one_error_naming_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(EnvFileError, r"^\.env is missing"):
                load_env(Path(temp), ".env")

    def test_a_malformed_file_is_one_error_naming_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("not an assignment\n", encoding="utf-8")
            with self.assertRaisesRegex(EnvFileError, r"^\.env: line 1"):
                load_env(root, ".env")


if __name__ == "__main__":
    unittest.main()

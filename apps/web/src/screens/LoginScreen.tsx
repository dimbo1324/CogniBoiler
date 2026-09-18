import { useState, type FormEvent } from "react";

import { ApiError } from "../api/http";
import { useSession } from "../session/SessionProvider";

/** What to tell the operator after a refused sign-in. Never whether the account exists. */
export function signInError(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.code) {
      case "auth.invalid_credentials":
        return "Invalid username or password.";
      case "auth.too_many_attempts": {
        const minutes = Math.max(1, Math.ceil((error.problem.retryAfterS ?? 60) / 60));
        return `Too many failed sign-in attempts. Try again in ${String(minutes)} min.`;
      }
      case "request.invalid":
        return "Enter a username of 3–64 characters and a password of at least 8.";
      case "network.unreachable":
        return "The gateway cannot be reached.";
      default:
        return error.problem.detail;
    }
  }
  return "Sign-in failed.";
}

export function LoginScreen({ notice }: { notice: string | null }) {
  const { manager } = useSession();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    manager.signIn(username.trim(), password).catch((failure: unknown) => {
      setError(signInError(failure));
      setPassword("");
      setBusy(false);
    });
  };

  return (
    <main className="login panel">
      <h1>CogniBoiler</h1>
      <p className="muted">Operator console of the 300 MW unit</p>
      {notice && (
        <p className="notice" role="status">
          {notice}
        </p>
      )}
      <form onSubmit={submit} aria-label="Sign in">
        <label>
          Username
          <input
            name="username"
            autoComplete="username"
            value={username}
            onChange={(event) => {
              setUsername(event.target.value);
            }}
            required
          />
        </label>
        <label>
          Password
          <input
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
            }}
            required
          />
        </label>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <button type="submit" className="primary" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}

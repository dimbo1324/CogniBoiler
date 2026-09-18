import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent, type ReactNode } from "react";

import {
  ROLES,
  createUser,
  fetchUsers,
  resetUserPassword,
  revokeUserSessions,
  updateUser,
} from "../api/endpoints";
import { describeError } from "../api/http";
import type { Role, User } from "../api/types";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Pager } from "../components/Pager";
import { useUser } from "../session/SessionProvider";
import { formatDateTime } from "../units";

const PAGE = 50;
// The gateway's password policy (schemas/auth.py).
export const PASSWORD_MIN_LENGTH = 12;
export const USERNAME_PATTERN = /^[A-Za-z0-9._-]{3,64}$/u;

interface PendingChange {
  title: string;
  body: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  run: () => Promise<unknown>;
  done: string;
}

function CreateUser({ onCreated }: { onCreated: (message: string) => void }) {
  const queryClient = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("viewer");
  const create = useMutation({
    mutationFn: () => createUser({ username: username.trim(), password, role }),
    onSuccess: (user) => {
      setUsername("");
      setPassword("");
      onCreated(`Created ${user.username} as ${user.role ?? "no role"}.`);
      return queryClient.invalidateQueries({ queryKey: ["users"] });
    },
  });
  const valid = USERNAME_PATTERN.test(username.trim()) && password.length >= PASSWORD_MIN_LENGTH;
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    create.mutate();
  };
  return (
    <section className="panel" aria-label="New user">
      <h2>New user</h2>
      <form className="form-grid" onSubmit={submit} aria-label="Create a user">
        <label>
          Username
          <input
            value={username}
            autoComplete="off"
            onChange={(event) => {
              setUsername(event.target.value);
            }}
          />
        </label>
        <label>
          Initial password
          <input
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
            }}
          />
          <span className="muted">at least {PASSWORD_MIN_LENGTH} characters</span>
        </label>
        <label>
          Role
          <select
            value={role}
            onChange={(event) => {
              setRole(event.target.value as Role);
            }}
          >
            {ROLES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" className="primary" disabled={!valid || create.isPending}>
          Create
        </button>
      </form>
      {create.error && (
        <p className="error" role="alert">
          {describeError(create.error)}
        </p>
      )}
    </section>
  );
}

function UserRow({
  user,
  self,
  ask,
}: {
  user: User;
  self: boolean;
  ask: (change: PendingChange) => void;
}) {
  const [role, setRole] = useState<Role>(user.role ?? "viewer");
  const [password, setPassword] = useState("");
  return (
    <tr data-testid={`user-${user.username}`} className={user.is_active ? "" : "severity-warning"}>
      <td>
        {user.username}
        {self && <span className="muted"> (you)</span>}
      </td>
      <td>
        <select
          aria-label={`Role of ${user.username}`}
          value={role}
          disabled={self}
          onChange={(event) => {
            setRole(event.target.value as Role);
          }}
        >
          {ROLES.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        {role !== user.role && (
          <button
            type="button"
            onClick={() => {
              ask({
                title: `Change the role of ${user.username}`,
                body: (
                  <p>
                    From {user.role ?? "no role"} to <strong>{role}</strong>. Their open sessions
                    close at once.
                  </p>
                ),
                confirmLabel: `Make ${role}`,
                run: () => updateUser(user.id, { role }),
                done: `${user.username} is now ${role}.`,
              });
            }}
          >
            Apply…
          </button>
        )}
      </td>
      <td>{user.is_active ? "active" : "blocked"}</td>
      <td>{formatDateTime(user.last_login_at_ms)}</td>
      <td className="number">{user.open_sessions}</td>
      <td>
        <div className="row">
          <button
            type="button"
            disabled={self}
            onClick={() => {
              ask({
                title: `${user.is_active ? "Block" : "Unblock"} ${user.username}`,
                body: user.is_active ? (
                  <p>They cannot sign in, and their open sessions close at once.</p>
                ) : (
                  <p>They can sign in again.</p>
                ),
                confirmLabel: user.is_active ? "Block" : "Unblock",
                danger: user.is_active,
                run: () => updateUser(user.id, { is_active: !user.is_active }),
                done: `${user.username} is ${user.is_active ? "blocked" : "active again"}.`,
              });
            }}
          >
            {user.is_active ? "Block…" : "Unblock…"}
          </button>
          <button
            type="button"
            disabled={user.open_sessions === 0}
            onClick={() => {
              ask({
                title: `Sign ${user.username} out everywhere`,
                body: <p>Every open session of {user.username} closes at once.</p>,
                confirmLabel: "Sign out everywhere",
                run: () => revokeUserSessions(user.id),
                done: `${user.username} was signed out everywhere.`,
              });
            }}
          >
            Sign out everywhere…
          </button>
          <input
            type="password"
            aria-label={`New password for ${user.username}`}
            placeholder="new password"
            autoComplete="new-password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
            }}
          />
          <button
            type="button"
            disabled={password.length < PASSWORD_MIN_LENGTH}
            onClick={() => {
              const chosen = password;
              ask({
                title: `Reset the password of ${user.username}`,
                body: <p>Their open sessions close; they sign in with the new password.</p>,
                confirmLabel: "Reset password",
                run: () =>
                  resetUserPassword(user.id, chosen).then((result) => {
                    setPassword("");
                    return result;
                  }),
                done: `The password of ${user.username} was reset.`,
              });
            }}
          >
            Reset password…
          </button>
        </div>
      </td>
    </tr>
  );
}

export function UsersScreen() {
  const me = useUser();
  const queryClient = useQueryClient();
  const [offset, setOffset] = useState(0);
  const [pending, setPending] = useState<PendingChange | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const users = useQuery({
    queryKey: ["users", offset],
    queryFn: ({ signal }) => fetchUsers(PAGE, offset, signal),
  });
  const change = useMutation({
    mutationFn: (run: () => Promise<unknown>) => run(),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["users"] }),
  });

  return (
    <div className="stack">
      <CreateUser onCreated={setMessage} />
      <section className="panel" aria-label="Users">
        <h2>Users</h2>
        {message && (
          <p className="ok-text" role="status">
            {message}
          </p>
        )}
        {change.error && (
          <p className="error" role="alert">
            {describeError(change.error)}
          </p>
        )}
        {users.isError && <p className="error">{describeError(users.error)}</p>}
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>User</th>
                <th>Role</th>
                <th>State</th>
                <th>Last sign-in</th>
                <th className="number">Sessions</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {(users.data?.items ?? []).map((user) => (
                <UserRow
                  key={`${String(user.id)}:${user.role ?? ""}:${String(user.is_active)}`}
                  user={user}
                  self={user.username === me.username}
                  ask={setPending}
                />
              ))}
            </tbody>
          </table>
        </div>
        <Pager offset={offset} limit={PAGE} total={users.data?.total ?? 0} onChange={setOffset} />
      </section>
      {pending && (
        <ConfirmDialog
          title={pending.title}
          confirmLabel={pending.confirmLabel}
          danger={pending.danger}
          busy={change.isPending}
          onCancel={() => {
            setPending(null);
          }}
          onConfirm={() => {
            const done = pending.done;
            setMessage(null);
            change.mutate(pending.run, {
              onSuccess: () => {
                setMessage(done);
              },
              onSettled: () => {
                setPending(null);
              },
            });
          }}
        >
          {pending.body}
        </ConfirmDialog>
      )}
    </div>
  );
}

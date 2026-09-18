import type { ReactNode } from "react";

import { useCan, useRole } from "../session/SessionProvider";
import type { Permission } from "../session/roles";

/**
 * A screen reached by its address without the role for it. The gateway would refuse its
 * requests anyway; this says so before the operator tries.
 */
export function RequirePermission({
  permission,
  children,
}: {
  permission: Permission;
  children: ReactNode;
}) {
  const allowed = useCan(permission);
  const role = useRole();
  if (!allowed) {
    return (
      <p className="notice" role="status">
        This screen is not available to the {role ?? "current"} role.
      </p>
    );
  }
  return <>{children}</>;
}

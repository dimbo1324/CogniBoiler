// The gateway's request bounds, in the SI units of the contract. They come from the
// gateway's pydantic schemas (schemas/command.py, schemas/plant.py, schemas/auth.py) through
// shared/openapi/api-gateway.json; the generated TypeScript types carry no bounds, so they
// are written here once and contract.test.ts checks every value against that JSON.

export const GATEWAY_LIMITS = {
  /** LoadDemandRequest.load_w [W] */
  loadW: [0, 300e6],
  /** SetpointRequest.pressure_pa [Pa] */
  pressurePa: [50e5, 185e5],
  /** SetpointRequest.water_level_m [m] */
  waterLevelM: [0.5, 9],
  /** SetpointRequest.steam_temp_k [K] */
  steamTempK: [400, 848],
  /** ValveCommandRequest.*_valve, a fraction of full travel */
  valveFraction: [0, 1],
  /** StepRequest.steps */
  steps: [1, 3600],
  /** FaultRequest.ramp_s [s] */
  faultRampS: [0, 3600],
  /** UserCreateRequest.password, PasswordResetRequest.new_password */
  passwordMinLength: 12,
  /** UserCreateRequest.username */
  usernamePattern: /^[A-Za-z0-9._-]{3,64}$/u,
} as const;

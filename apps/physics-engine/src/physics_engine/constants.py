"""Physical constants for the CogniBoiler steam boiler model."""

# ─── Thermodynamic constants ────────────────────────────────────────────────
SPECIFIC_HEAT_WATER: float = 4186.0  # J/(kg·K)  — specific heat of liquid water
SPECIFIC_HEAT_STEAM: float = 2010.0  # J/(kg·K)  — specific heat of superheated steam
LATENT_HEAT_VAPORIZATION: float = (
    2.26e6  # J/kg      — heat of vaporization at ~100°C (reference)
)
WATER_DENSITY: float = 850.0  # kg/m³     — density of hot pressurized water (~300°C)

# ─── Boiler drum geometry ───────────────────────────────────────────────────
DRUM_VOLUME: float = 50.0  # m³        — total drum volume
DRUM_CROSS_SECTION: float = 6.0  # m²        — cross-sectional area of drum
DRUM_HEIGHT: float = 8.0  # m         — total drum height

# ─── Operating ranges ───────────────────────────────────────────────────────
PRESSURE_MIN: float = 20.0e5  # Pa — minimum operating pressure (20 bar, safety floor)
PRESSURE_MAX: float = 180.0e5  # Pa        — maximum operating pressure (180 bar)
PRESSURE_NOMINAL: float = 140.0e5  # Pa        — nominal operating pressure (140 bar)

TEMP_STEAM_MIN: float = 813.15  # K         — min superheated steam temp (540°C)
TEMP_STEAM_MAX: float = 838.15  # K         — max superheated steam temp (565°C)
TEMP_STEAM_NOMINAL: float = 825.65  # K         — nominal steam temp (552.5°C)

TEMP_FEEDWATER: float = 423.15  # K         — feedwater inlet temperature (150°C)
TEMP_AMBIENT: float = 293.15  # K         — ambient temperature (20°C)

# ─── Combustion ─────────────────────────────────────────────────────────────
FUEL_HEATING_VALUE: float = 42.0e6  # J/kg      — lower heating value of natural gas
COMBUSTION_EFFICIENCY: float = 0.92  # —         — combustion efficiency (92%)
MAX_FUEL_FLOW: float = 10.0  # kg/s      — maximum fuel mass flow rate

# ─── Heat transfer ───────────────────────────────────────────────────────────
HEAT_LOSS_COEFFICIENT: float = 500.0  # W/K       — overall heat loss coefficient (UA)

# FIX (v3): previous values were 15 000 W/K and 250 000 W/K — both insufficient.
#
# At full feedwater (300 kg/s, 150 °C) flowing into a 337 °C drum, the
# cold-dilution power demand is:
#   P_dil = 300 × 5 000 × (610 − 423) K ≈ 281 MW
#
# With UA = 250 000 W/K combustion only delivers 250 000 × 663 ≈ 166 MW < 281 MW
# -> temperature (and hence pressure) fell even under full firing.
#
# With UA = 450 000 W/K:
#   Q_gas = 450 000 × (1273 − 610) = 298 MW > 281 MW  -> temperature rises  ✓
#
# Physical sanity: a 300 t/h utility boiler has ~3 000 m² heating surface and
# gas-side HTC ≈ 120–160 W/(m²·K) -> UA ≈ 3 000 × 150 = 450 000 W/K.
HEAT_TRANSFER_GAS_WATER: float = 450_000.0  # W/K — gas-to-water UA

# ─── Steam flow ─────────────────────────────────────────────────────────────
STEAM_VALVE_COEFFICIENT: float = 50.0  # kg/(s·bar) — steam valve flow coefficient
MAX_STEAM_FLOW: float = 277.8  # kg/s       — max steam flow (1000 t/h)
MIN_STEAM_FLOW: float = 138.9  # kg/s       — min steam flow (500 t/h)

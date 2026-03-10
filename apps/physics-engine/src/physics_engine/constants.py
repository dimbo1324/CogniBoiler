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
PRESSURE_MIN: float = 100.0e5  # Pa        — minimum operating pressure (100 bar)
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
HEAT_TRANSFER_GAS_WATER: float = (
    15000.0  # W/K     — gas-to-water heat transfer coefficient
)

# ─── Steam flow ─────────────────────────────────────────────────────────────
STEAM_VALVE_COEFFICIENT: float = 50.0  # kg/(s·bar) — steam valve flow coefficient
MAX_STEAM_FLOW: float = 277.8  # kg/s       — max steam flow (1000 t/h)
MIN_STEAM_FLOW: float = 138.9  # kg/s       — min steam flow (500 t/h)

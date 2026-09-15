"""Physical constants and design data of the CogniBoiler 300 MW gas-fired drum unit."""

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
NOMINAL_WATER_LEVEL: float = 4.8  # m   — normal water level, 60 % of drum height

# ─── Operating ranges ───────────────────────────────────────────────────────
PRESSURE_MIN: float = 20.0e5  # Pa — minimum operating pressure (20 bar, safety floor)
PRESSURE_MAX: float = 180.0e5  # Pa        — maximum operating pressure (180 bar)
PRESSURE_NOMINAL: float = 140.0e5  # Pa        — nominal operating pressure (140 bar)

TEMP_STEAM_MIN: float = 813.15  # K         — min superheated steam temp (540°C)
TEMP_STEAM_MAX: float = 838.15  # K         — max superheated steam temp (565°C)
TEMP_STEAM_NOMINAL: float = 825.65  # K         — nominal steam temp (552.5°C)
TEMP_STEAM_RATED: float = 811.0  # K — turbine inlet design temperature after spray

TEMP_FEEDWATER: float = 423.15  # K         — feedwater inlet temperature (150°C)
TEMP_AMBIENT: float = 293.15  # K         — ambient temperature (20°C)

# ─── Unit rating ────────────────────────────────────────────────────────────
RATED_POWER: float = 300.0e6  # W — maximum continuous electrical output
NOMINAL_LOAD: float = 250.0e6  # W — the load the plant starts at by default
RATED_STEAM_FLOW: float = 245.0  # kg/s — turbine steam flow at rated power

# ─── Combustion ─────────────────────────────────────────────────────────────
FUEL_HEATING_VALUE: float = 42.0e6  # J/kg      — lower heating value of natural gas
COMBUSTION_EFFICIENCY: float = 0.92  # —         — combustion efficiency (92%)
# Rated firing is ~19.5 kg/s (~820 MW of fuel): the fuel valve sits near 0.78 at
# 300 MW, leaving control margin. 10 kg/s could not raise the steam a 300 MW turbine
# needs without the model creating energy.
MAX_FUEL_FLOW: float = 25.0  # kg/s      — maximum fuel mass flow rate

# ─── Heat transfer ───────────────────────────────────────────────────────────
HEAT_LOSS_COEFFICIENT: float = 500.0  # W/K       — overall heat loss coefficient (UA)

# Furnace water walls: the radiant evaporator. Sized from the rated heat balance —
# about 280 MW absorbed with furnace exit gas at 1 400 K and saturated water at 610 K.
# The rest of the evaporation happens in the convective evaporator bank.
HEAT_TRANSFER_GAS_WATER: float = 351_000.0  # W/K — furnace gas-to-water UA

# Heat stored in the pressure parts and in the water of the downcomers and water walls,
# outside the drum section whose level is modelled. It sets how fast drum pressure moves
# when firing and steam demand disagree: about 6 bar/min for a 10 % mismatch at rated
# load, the order of magnitude of a real 300 MW drum unit.
BOILER_STORAGE_HEAT_CAPACITY: float = 2.0e8  # J/K

# ─── Steam, feedwater and spray flow ────────────────────────────────────────
# Turbine admission is choked: flow is proportional to valve opening and drum pressure.
# Rated flow at nominal pressure needs the valve about 85 % open.
STEAM_VALVE_COEFFICIENT: float = 290.0  # kg/s — fully open valve at nominal pressure
MAX_STEAM_FLOW: float = 277.8  # kg/s       — max steam flow (1000 t/h)
MIN_STEAM_FLOW: float = 138.9  # kg/s       — min steam flow (500 t/h)
MAX_FEEDWATER_FLOW: float = 380.0  # kg/s — feedwater valve fully open, pump healthy
MAX_SPRAY_FLOW: float = 30.0  # kg/s — attemperator spray valve fully open
# Spray is capped to this fraction of the steam flow: more water than that would reach
# the turbine unevaporated.
SPRAY_MAX_FRACTION: float = 0.15

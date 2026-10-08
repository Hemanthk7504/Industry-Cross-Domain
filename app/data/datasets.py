import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Domain 1: Source Domain (HVAC Operating Telemetry)
HVAC_FEATURE_NAMES = [
    "indoor_temp",
    "return_air_temp",
    "supply_air_temp",
    "chilled_water_supply_temp",
    "chilled_water_return_temp",
    "fan_power",
    "compressor_vibration",
    "air_flow_rate",
    "static_pressure",
    "relative_humidity"
]

HVAC_FEATURE_METADATA = {
    "indoor_temp": {"name": "Indoor Zone Temp", "unit": "°C", "min": 18.0, "max": 28.0, "normal": 22.5, "desc": "Conditioned zone ambient air temperature"},
    "return_air_temp": {"name": "Return Air Temp", "unit": "°C", "min": 20.0, "max": 30.0, "normal": 24.2, "desc": "Temperature of recirculated air stream"},
    "supply_air_temp": {"name": "Supply Air Temp", "unit": "°C", "min": 11.0, "max": 18.0, "normal": 14.5, "desc": "Discharge air temperature from air handling unit"},
    "chilled_water_supply_temp": {"name": "Chilled Water Supply", "unit": "°C", "min": 5.0, "max": 10.0, "normal": 6.8, "desc": "Chiller loop leaving water temperature"},
    "chilled_water_return_temp": {"name": "Chilled Water Return", "unit": "°C", "min": 10.0, "max": 16.0, "normal": 12.4, "desc": "Chiller loop entering water temperature"},
    "fan_power": {"name": "Supply Fan Power", "unit": "kW", "min": 1.5, "max": 16.0, "normal": 7.2, "desc": "Electric motor active power consumption"},
    "compressor_vibration": {"name": "Compressor Vibration", "unit": "mm/s", "min": 0.5, "max": 8.0, "normal": 1.8, "desc": "Chiller scroll/centrifugal compressor vibration velocity"},
    "air_flow_rate": {"name": "Volumetric Air Flow", "unit": "m³/h", "min": 1200.0, "max": 4800.0, "normal": 3200.0, "desc": "Total AHU supply air volumetric throughput"},
    "static_pressure": {"name": "Duct Static Pressure", "unit": "Pa", "min": 120.0, "max": 580.0, "normal": 320.0, "desc": "Air distribution static pressure head"},
    "relative_humidity": {"name": "Return Humidity", "unit": "%", "min": 30.0, "max": 80.0, "normal": 50.0, "desc": "Relative humidity in return plenum"}
}

# Domain 2: Target Domain (Predictive Maintenance Telemetry)
MAINTENANCE_FEATURE_NAMES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "vibration_index",
    "acoustic_emission",
    "oil_pressure"
]

MAINTENANCE_FEATURE_METADATA = {
    "air_temperature": {"name": "Air Temperature", "unit": "K", "min": 295.0, "max": 305.0, "normal": 298.1, "desc": "Ambient shop floor thermal boundary"},
    "process_temperature": {"name": "Process Temperature", "unit": "K", "min": 305.0, "max": 315.0, "normal": 308.6, "desc": "Internal spindle/casing interface temperature"},
    "rotational_speed": {"name": "Rotational Speed", "unit": "rpm", "min": 1100.0, "max": 2880.0, "normal": 1540.0, "desc": "Spindle drive shaft rotational velocity"},
    "torque": {"name": "Spindle Torque", "unit": "Nm", "min": 10.0, "max": 80.0, "normal": 40.2, "desc": "Electromechanical torque output on cutting tool"},
    "tool_wear": {"name": "Tool Wear Duration", "unit": "min", "min": 0.0, "max": 250.0, "normal": 85.0, "desc": "Cumulative cutting contact wear time"},
    "vibration_index": {"name": "Mechanical Vibration", "unit": "mm/s", "min": 0.5, "max": 10.0, "normal": 2.1, "desc": "Root-mean-square multi-axis bearing vibration"},
    "acoustic_emission": {"name": "Acoustic Emission", "unit": "dB", "min": 45.0, "max": 98.0, "normal": 62.0, "desc": "High-frequency micro-cracking acoustic energy"},
    "oil_pressure": {"name": "Lube Oil Pressure", "unit": "bar", "min": 2.0, "max": 7.0, "normal": 4.5, "desc": "Hydraulic lubrication feed line pressure"}
}

FAILURE_MODES = {
    "TWF": {
        "title": "Tool Wear Failure",
        "description": "Critical abrasive tip wear exceeding structural micro-cutting tolerances (200-240 min).",
        "severity": "CRITICAL",
        "action": "Immediate spindle stop. Replace carbide cutting insert and verify tool offset calibration."
    },
    "HDF": {
        "title": "Heat Dissipation Failure",
        "description": "Thermal differential between process and ambient air collapsed under 8.6 K at low speed.",
        "severity": "HIGH",
        "action": "Flush coolant supply conduits, inspect external cooling radiator, reduce feed rate by 30%."
    },
    "PWF": {
        "title": "Power Overload/Stall Failure",
        "description": "Calculated mechanical power (Torque × Speed × 2π / 60) exceeded 9,000 W or dropped below 3,500 W under load.",
        "severity": "CRITICAL",
        "action": "Check motor inverter drive, inspect gear mesh friction, reset thermal circuit protection."
    },
    "OSF": {
        "title": "Overstrain Failure",
        "description": "Torque and tool wear combined product exceeds machine structural strain limits (>11,000 Nm·min).",
        "severity": "EMERGENCY",
        "action": "Halt machining program immediately. Perform spindle run-out test and check workpiece clamping torque."
    },
    "RNF": {
        "title": "Random Electro-Mechanical Anomaly",
        "description": "Transient high-frequency harmonic vibration or pressure cavitation spike detected.",
        "severity": "MODERATE",
        "action": "Perform ultrasonic lubrication check and run diagnostic self-test sequence."
    }
}

def generate_synthetic_hvac_data(n_samples: int = 2000, anomaly_ratio: float = 0.12, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic HVAC operational source domain data with realistic thermodynamics."""
    np.random.seed(random_state)
    n_anom = int(n_samples * anomaly_ratio)
    n_norm = n_samples - n_anom
    
    # Normal operating baseline
    norm_indoor = np.random.normal(22.5, 1.2, n_norm)
    norm_return = norm_indoor + np.random.normal(1.8, 0.4, n_norm)
    norm_supply = np.random.normal(14.5, 0.8, n_norm)
    norm_chilled_sup = np.random.normal(6.8, 0.5, n_norm)
    norm_chilled_ret = norm_chilled_sup + np.random.normal(5.6, 0.6, n_norm)
    norm_fan_power = np.random.normal(7.2, 1.1, n_norm)
    norm_vib = np.random.gamma(2.0, 0.8, n_norm)
    norm_flow = np.random.normal(3200, 250, n_norm)
    norm_pres = np.random.normal(320, 25, n_norm)
    norm_rh = np.random.normal(50.0, 5.0, n_norm)
    
    X_norm = np.column_stack([
        norm_indoor, norm_return, norm_supply, norm_chilled_sup, norm_chilled_ret,
        norm_fan_power, norm_vib, norm_flow, norm_pres, norm_rh
    ])
    y_norm = np.zeros(n_norm)
    
    # Anomalous data (chiller fouling, damper stuck, heat wave stress)
    anom_indoor = np.random.uniform(25.5, 28.5, n_anom)
    anom_return = anom_indoor + np.random.uniform(2.5, 4.0, n_anom)
    anom_supply = np.random.uniform(17.5, 21.0, n_anom)
    anom_chilled_sup = np.random.uniform(8.5, 11.5, n_anom)
    anom_chilled_ret = anom_chilled_sup + np.random.uniform(2.0, 4.0, n_anom)
    anom_fan_power = np.random.uniform(12.0, 16.5, n_anom)
    anom_vib = np.random.uniform(4.5, 8.5, n_anom)
    anom_flow = np.random.uniform(1200, 2200, n_anom)
    anom_pres = np.random.uniform(450, 580, n_anom)
    anom_rh = np.random.uniform(65, 85, n_anom)
    
    X_anom = np.column_stack([
        anom_indoor, anom_return, anom_supply, anom_chilled_sup, anom_chilled_ret,
        anom_fan_power, anom_vib, anom_flow, anom_pres, anom_rh
    ])
    y_anom = np.ones(n_anom)
    
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([y_norm, y_anom])
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

def generate_synthetic_maintenance_data(n_samples: int = 3000, failure_ratio: float = 0.08, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generates synthetic predictive maintenance target domain data based on AI4I industrial standards."""
    np.random.seed(random_state)
    n_fail = int(n_samples * failure_ratio)
    n_norm = n_samples - n_fail
    
    # Normal target observations
    air_temp = np.random.normal(298.1, 1.8, n_norm)
    proc_temp = air_temp + np.random.normal(10.5, 0.8, n_norm)
    rot_speed = np.random.normal(1540.0, 140.0, n_norm)
    torque = np.random.normal(40.2, 8.5, n_norm)
    tool_wear = np.random.uniform(0.0, 175.0, n_norm)
    vib = np.random.gamma(2.2, 0.9, n_norm)
    acoustic = np.random.normal(62.0, 4.5, n_norm)
    oil_press = np.random.normal(4.5, 0.5, n_norm)
    
    X_norm = np.column_stack([
        air_temp, proc_temp, rot_speed, torque, tool_wear, vib, acoustic, oil_press
    ])
    y_norm = np.zeros(n_norm)
    fail_modes_norm = ["None"] * n_norm
    
    # Failure observations (distributed across 5 physics-based failure mechanisms)
    n_per_mode = n_fail // 5
    
    # TWF (Tool Wear Failure)
    twf_tool_wear = np.random.uniform(200.0, 245.0, n_per_mode)
    twf_torque = np.random.normal(52.0, 6.0, n_per_mode)
    twf_rot = np.random.normal(1420.0, 120.0, n_per_mode)
    twf_air = np.random.normal(299.0, 1.5, n_per_mode)
    twf_proc = twf_air + np.random.normal(11.0, 0.7, n_per_mode)
    twf_vib = np.random.uniform(4.5, 7.5, n_per_mode)
    twf_ac = np.random.uniform(78.0, 92.0, n_per_mode)
    twf_oil = np.random.normal(4.2, 0.6, n_per_mode)
    X_twf = np.column_stack([twf_air, twf_proc, twf_rot, twf_torque, twf_tool_wear, twf_vib, twf_ac, twf_oil])
    
    # HDF (Heat Dissipation Failure: temp difference < 8.6K and rot_speed < 1380 rpm)
    hdf_air = np.random.uniform(301.0, 304.5, n_per_mode)
    hdf_proc = hdf_air + np.random.uniform(6.5, 8.2, n_per_mode)  # small delta
    hdf_rot = np.random.uniform(1150.0, 1370.0, n_per_mode)
    hdf_torque = np.random.uniform(55.0, 72.0, n_per_mode)
    hdf_tool = np.random.uniform(60.0, 180.0, n_per_mode)
    hdf_vib = np.random.uniform(3.5, 6.0, n_per_mode)
    hdf_ac = np.random.normal(68.0, 5.0, n_per_mode)
    hdf_oil = np.random.uniform(2.5, 3.8, n_per_mode)
    X_hdf = np.column_stack([hdf_air, hdf_proc, hdf_rot, hdf_torque, hdf_tool, hdf_vib, hdf_ac, hdf_oil])
    
    # PWF (Power Failure: power out of safe operating envelope)
    pwf_rot = np.random.choice([np.random.uniform(1150, 1280), np.random.uniform(2400, 2800)], size=n_per_mode)
    pwf_torque = np.random.uniform(62.0, 78.0, n_per_mode)
    pwf_air = np.random.normal(298.5, 1.8, n_per_mode)
    pwf_proc = pwf_air + np.random.normal(10.5, 0.8, n_per_mode)
    pwf_tool = np.random.uniform(40.0, 190.0, n_per_mode)
    pwf_vib = np.random.uniform(5.0, 8.5, n_per_mode)
    pwf_ac = np.random.uniform(75.0, 95.0, n_per_mode)
    pwf_oil = np.random.uniform(2.2, 3.5, n_per_mode)
    X_pwf = np.column_stack([pwf_air, pwf_proc, pwf_rot, pwf_torque, pwf_tool, pwf_vib, pwf_ac, pwf_oil])
    
    # OSF (Overstrain Failure: torque * tool_wear > 11000)
    osf_tool = np.random.uniform(160.0, 230.0, n_per_mode)
    osf_torque = np.random.uniform(60.0, 78.0, n_per_mode)
    osf_rot = np.random.normal(1350.0, 100.0, n_per_mode)
    osf_air = np.random.normal(299.2, 1.6, n_per_mode)
    osf_proc = osf_air + np.random.normal(11.2, 0.9, n_per_mode)
    osf_vib = np.random.uniform(6.0, 9.5, n_per_mode)
    osf_ac = np.random.uniform(80.0, 96.0, n_per_mode)
    osf_oil = np.random.uniform(2.1, 3.2, n_per_mode)
    X_osf = np.column_stack([osf_air, osf_proc, osf_rot, osf_torque, osf_tool, osf_vib, osf_ac, osf_oil])
    
    # RNF (Random Failure)
    rnf_samples = n_fail - (4 * n_per_mode)
    rnf_air = np.random.normal(298.1, 2.0, rnf_samples)
    rnf_proc = rnf_air + np.random.normal(10.5, 1.0, rnf_samples)
    rnf_rot = np.random.normal(1500.0, 180.0, rnf_samples)
    rnf_torque = np.random.normal(48.0, 12.0, rnf_samples)
    rnf_tool = np.random.uniform(80.0, 200.0, rnf_samples)
    rnf_vib = np.random.uniform(7.0, 10.0, rnf_samples)
    rnf_ac = np.random.uniform(85.0, 98.0, rnf_samples)
    rnf_oil = np.random.uniform(2.0, 3.0, rnf_samples)
    X_rnf = np.column_stack([rnf_air, rnf_proc, rnf_rot, rnf_torque, rnf_tool, rnf_vib, rnf_ac, rnf_oil])
    
    X_fail = np.vstack([X_twf, X_hdf, X_pwf, X_osf, X_rnf])
    y_fail = np.ones(len(X_fail))
    fail_modes_fail = ["TWF"] * n_per_mode + ["HDF"] * n_per_mode + ["PWF"] * n_per_mode + ["OSF"] * n_per_mode + ["RNF"] * rnf_samples
    
    X = np.vstack([X_norm, X_fail])
    y = np.concatenate([y_norm, y_fail])
    fail_modes = fail_modes_norm + fail_modes_fail
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices], [fail_modes[i] for i in indices]

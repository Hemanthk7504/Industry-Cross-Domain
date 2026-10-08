from typing import Dict, Any

INDUSTRIAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "normal_nominal": {
        "id": "normal_nominal",
        "title": "Nominal Baseline (Asset CNC-402)",
        "badge": "Healthy Operation",
        "badge_color": "emerald",
        "description": "Asset running under balanced machining loads with optimal lubrication, sharp tool tip, and normal heat dissipation.",
        "target_sensors": {
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1550.0,
            "torque": 40.2,
            "tool_wear": 42.0,
            "vibration_index": 1.85,
            "acoustic_emission": 58.2,
            "oil_pressure": 4.65
        },
        "source_hvac_sensors": {
            "indoor_temp": 22.4,
            "return_air_temp": 24.1,
            "supply_air_temp": 14.5,
            "chilled_water_supply_temp": 6.8,
            "chilled_water_return_temp": 12.3,
            "fan_power": 6.9,
            "compressor_vibration": 1.6,
            "air_flow_rate": 3250.0,
            "static_pressure": 315.0,
            "relative_humidity": 48.0
        },
        "expected_prediction": "Normal",
        "expected_failure_mode": "None",
        "expected_confidence": 0.994
    },
    "tool_wear_critical": {
        "id": "tool_wear_critical",
        "title": "Critical Tool Wear (Asset CNC-108)",
        "badge": "TWF Impending",
        "badge_color": "rose",
        "description": "Carbide milling insert has surpassed 218 minutes of active cutting time; high friction causes acoustic emission spikes and high torque.",
        "target_sensors": {
            "air_temperature": 299.4,
            "process_temperature": 310.8,
            "rotational_speed": 1395.0,
            "torque": 56.4,
            "tool_wear": 224.0,
            "vibration_index": 5.40,
            "acoustic_emission": 86.5,
            "oil_pressure": 4.10
        },
        "source_hvac_sensors": {
            "indoor_temp": 24.8,
            "return_air_temp": 27.2,
            "supply_air_temp": 16.2,
            "chilled_water_supply_temp": 8.1,
            "chilled_water_return_temp": 13.9,
            "fan_power": 9.8,
            "compressor_vibration": 3.4,
            "air_flow_rate": 2800.0,
            "static_pressure": 380.0,
            "relative_humidity": 58.0
        },
        "expected_prediction": "Imminent Failure",
        "expected_failure_mode": "TWF",
        "expected_confidence": 0.988
    },
    "heat_dissipation_failure": {
        "id": "heat_dissipation_failure",
        "title": "Cooling System Choked (Asset LATHE-09)",
        "badge": "HDF Risk",
        "badge_color": "amber",
        "description": "Temperature delta between process chamber (309.2 K) and ambient air (302.5 K) collapsed to only 6.7 K under low rotational speed (1240 rpm).",
        "target_sensors": {
            "air_temperature": 302.5,
            "process_temperature": 309.2,
            "rotational_speed": 1240.0,
            "torque": 64.8,
            "tool_wear": 142.0,
            "vibration_index": 4.15,
            "acoustic_emission": 71.0,
            "oil_pressure": 3.10
        },
        "source_hvac_sensors": {
            "indoor_temp": 27.5,
            "return_air_temp": 30.2,
            "supply_air_temp": 19.8,
            "chilled_water_supply_temp": 10.5,
            "chilled_water_return_temp": 14.8,
            "fan_power": 14.2,
            "compressor_vibration": 5.8,
            "air_flow_rate": 1850.0,
            "static_pressure": 490.0,
            "relative_humidity": 72.0
        },
        "expected_prediction": "Imminent Failure",
        "expected_failure_mode": "HDF",
        "expected_confidence": 0.976
    },
    "overstrain_failure": {
        "id": "overstrain_failure",
        "title": "Severe Mechanical Overstrain (Asset MILL-33)",
        "badge": "OSF Emergency",
        "badge_color": "red",
        "description": "High torque (71.5 Nm) coupled with tool wear (195 min) pushes structural stress product to 13,942 Nm·min (threshold: 11,000 Nm·min).",
        "target_sensors": {
            "air_temperature": 299.8,
            "process_temperature": 311.2,
            "rotational_speed": 1310.0,
            "torque": 71.5,
            "tool_wear": 195.0,
            "vibration_index": 7.80,
            "acoustic_emission": 89.2,
            "oil_pressure": 2.75
        },
        "source_hvac_sensors": {
            "indoor_temp": 26.2,
            "return_air_temp": 29.0,
            "supply_air_temp": 17.5,
            "chilled_water_supply_temp": 9.4,
            "chilled_water_return_temp": 14.1,
            "fan_power": 12.5,
            "compressor_vibration": 6.2,
            "air_flow_rate": 2100.0,
            "static_pressure": 460.0,
            "relative_humidity": 66.0
        },
        "expected_prediction": "Imminent Failure",
        "expected_failure_mode": "OSF",
        "expected_confidence": 0.997
    },
    "power_failure": {
        "id": "power_failure",
        "title": "Inverter Power Stall/Overdrive (Asset PUMP-12)",
        "badge": "PWF Alert",
        "badge_color": "orange",
        "description": "Spindle motor delivering 74.2 Nm at high speed 2620 rpm resulting in mechanical power of 20,350 W, far beyond the 9,000 W thermal envelope.",
        "target_sensors": {
            "air_temperature": 298.9,
            "process_temperature": 309.8,
            "rotational_speed": 2620.0,
            "torque": 74.2,
            "tool_wear": 90.0,
            "vibration_index": 6.90,
            "acoustic_emission": 82.4,
            "oil_pressure": 2.90
        },
        "source_hvac_sensors": {
            "indoor_temp": 25.1,
            "return_air_temp": 27.8,
            "supply_air_temp": 16.0,
            "chilled_water_supply_temp": 8.5,
            "chilled_water_return_temp": 13.5,
            "fan_power": 11.2,
            "compressor_vibration": 4.9,
            "air_flow_rate": 2450.0,
            "static_pressure": 410.0,
            "relative_humidity": 60.0
        },
        "expected_prediction": "Imminent Failure",
        "expected_failure_mode": "PWF",
        "expected_confidence": 0.985
    }
}

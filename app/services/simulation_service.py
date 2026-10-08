import numpy as np
import time
from typing import Dict, Any, List

class SimulationService:
    """Simulates real-time industrial digital twin sensor telemetry and degradation streams."""

    def __init__(self):
        self.fleet_assets = [
            {"id": "CNC-402", "name": "5-Axis High-Speed Machining Center", "zone": "Bay 1", "status": "HEALTHY", "health_index": 98.4, "risk": "Low", "risk_color": "emerald"},
            {"id": "CNC-108", "name": "Precision CNC Milling Spindle", "zone": "Bay 1", "status": "WARNING", "health_index": 62.1, "risk": "Elevated", "risk_color": "amber"},
            {"id": "LATHE-09", "name": "Horizontal Turning Center", "zone": "Bay 2", "status": "HEALTHY", "health_index": 94.8, "risk": "Low", "risk_color": "emerald"},
            {"id": "MILL-33", "name": "Heavy Gantry Bed Milling Machine", "zone": "Bay 3", "status": "CRITICAL", "health_index": 18.2, "risk": "Imminent Failure", "risk_color": "rose"},
            {"id": "PUMP-12", "name": "High-Pressure Hydraulic Chiller Pump", "zone": "Utility 1", "status": "HEALTHY", "health_index": 91.5, "risk": "Low", "risk_color": "emerald"},
            {"id": "AHU-CHILL-01", "name": "Source HVAC Central Chiller Unit", "zone": "Utility Roof", "status": "HEALTHY", "health_index": 96.0, "risk": "Low", "risk_color": "emerald"}
        ]

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Returns plant-wide fleet telemetry overview."""
        total = len(self.fleet_assets)
        healthy = sum(1 for a in self.fleet_assets if a["status"] == "HEALTHY")
        warning = sum(1 for a in self.fleet_assets if a["status"] == "WARNING")
        critical = sum(1 for a in self.fleet_assets if a["status"] == "CRITICAL")
        
        return {
            "total_assets": total,
            "healthy_assets": healthy,
            "warning_assets": warning,
            "critical_assets": critical,
            "fleet_availability_pct": round(((healthy + warning * 0.7) / total) * 100, 1),
            "assets": self.fleet_assets
        }

    def generate_streaming_cycle(self, asset_id: str, cycle_step: int, degradation_mode: str = "normal") -> Dict[str, Any]:
        """
        Generates simulated live sensor telemetry stream point with progressive drift.
        Modes: 'normal', 'wear_drift', 'thermal_collapse', 'vibration_surge'.
        """
        t = cycle_step % 60
        
        # Base normal values with micro-noise
        base_air = 298.1 + 0.3 * np.sin(t / 5.0) + np.random.normal(0, 0.08)
        base_proc = base_air + 10.5 + 0.2 * np.cos(t / 4.0) + np.random.normal(0, 0.1)
        base_speed = 1540.0 + 25.0 * np.sin(t / 3.0) + np.random.normal(0, 5.0)
        base_torque = 40.2 + 2.5 * np.cos(t / 6.0) + np.random.normal(0, 0.4)
        base_wear = 45.0 + (cycle_step * 0.8)
        base_vib = 1.9 + 0.2 * np.sin(t / 2.0) + np.random.normal(0, 0.05)
        base_acous = 62.0 + np.random.normal(0, 1.2)
        base_oil = 4.5 + np.random.normal(0, 0.04)

        if degradation_mode == "wear_drift":
            # Tool wear accelerating towards 220 min
            base_wear = min(240.0, 140.0 + (cycle_step * 2.5))
            base_torque += (base_wear / 240.0) * 18.0
            base_vib += (base_wear / 240.0) * 3.5
            base_acous += (base_wear / 240.0) * 22.0
            health = max(10.0, 100.0 - (base_wear / 2.4))
        elif degradation_mode == "thermal_collapse":
            # Process & air temperature delta collapses under 8.6K
            base_air += min(6.0, cycle_step * 0.15)
            base_proc = base_air + max(5.8, 10.5 - (cycle_step * 0.12))
            base_speed = max(1180.0, 1540.0 - (cycle_step * 8.0))
            base_torque += min(25.0, cycle_step * 0.6)
            health = max(12.0, 100.0 - (cycle_step * 1.8))
        elif degradation_mode == "vibration_surge":
            # Sub-harmonic mechanical bearing fatigue
            base_vib += min(6.5, cycle_step * 0.22)
            base_acous += min(28.0, cycle_step * 0.8)
            base_oil = max(2.2, 4.5 - (cycle_step * 0.05))
            health = max(15.0, 100.0 - (cycle_step * 1.6))
        else:
            health = 98.4 - (base_wear / 50.0)

        # Stage 1 VAE anomaly score simulation
        vae_score = min(0.99, max(0.02, (100.0 - health) / 90.0))
        
        # Stage 2 BiLSTM-BiGRU failure probability
        fail_prob = min(0.999, max(0.005, 1.0 / (1.0 + np.exp(-( (100.0 - health) - 45.0 ) / 8.0))))

        return {
            "cycle_step": cycle_step,
            "timestamp": time.strftime("%H:%M:%S"),
            "asset_id": asset_id,
            "degradation_mode": degradation_mode,
            "health_index": round(float(health), 1),
            "sensors": {
                "air_temperature": round(float(base_air), 2),
                "process_temperature": round(float(base_proc), 2),
                "temp_delta": round(float(base_proc - base_air), 2),
                "rotational_speed": round(float(base_speed), 1),
                "torque": round(float(base_torque), 1),
                "tool_wear": round(float(base_wear), 1),
                "vibration_index": round(float(base_vib), 2),
                "acoustic_emission": round(float(base_acous), 1),
                "oil_pressure": round(float(base_oil), 2)
            },
            "stage1_vae_anomaly_score": round(float(vae_score), 4),
            "stage2_failure_probability": round(float(fail_prob), 4),
            "is_alert": bool(fail_prob >= 0.38)
        }

simulation_service = SimulationService()

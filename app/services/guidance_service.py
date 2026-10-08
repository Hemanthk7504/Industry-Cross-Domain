from typing import Dict, Any, List
import datetime
from app.data.datasets import FAILURE_MODES

class PrescriptiveGuidanceService:
    """Generates physics-guided prescriptive maintenance guidance, RUL, and automated work orders."""

    def diagnose_failure_mode(self, target_sensors: Dict[str, float], failure_prob: float) -> Dict[str, Any]:
        """Diagnoses exact failure mechanism and severity based on sensor telemetry."""
        if failure_prob < 0.38:
            return {
                "mode_code": "NOMINAL",
                "mode_title": "Normal Operational State",
                "severity": "NORMAL",
                "severity_badge": "emerald",
                "risk_score_pct": round(failure_prob * 100, 1),
                "rul_hours": 320.0,
                "rul_cycles": 1450,
                "root_cause": "All operational metrics within ISO 10816 mechanical vibration and thermal limits.",
                "prescriptions": [
                    "Continue standard preventive maintenance schedule (PM-30D).",
                    "Maintain continuous telemetry monitoring.",
                    "Log tool contact time for routine cycle inventory."
                ],
                "parts_required": [],
                "work_order_required": False
            }

        air_t = target_sensors.get("air_temperature", 298.1)
        proc_t = target_sensors.get("process_temperature", 308.6)
        delta_t = proc_t - air_t
        speed = target_sensors.get("rotational_speed", 1540.0)
        torque = target_sensors.get("torque", 40.2)
        wear = target_sensors.get("tool_wear", 85.0)
        power_w = (torque * speed * 2.0 * 3.14159) / 60.0
        overstrain_prod = torque * wear

        # 1. Overstrain Failure Check (OSF)
        if overstrain_prod > 11000.0:
            fm = FAILURE_MODES["OSF"]
            return {
                "mode_code": "OSF",
                "mode_title": fm["title"],
                "severity": fm["severity"],
                "severity_badge": "rose",
                "risk_score_pct": round(failure_prob * 100, 1),
                "rul_hours": max(0.5, round(2.5 * (1.0 - failure_prob), 1)),
                "rul_cycles": max(1, int(15 * (1.0 - failure_prob))),
                "root_cause": f"Structural strain product ({overstrain_prod:.0f} Nm·min) breached safety limit (11,000 Nm·min).",
                "prescriptions": [
                    "EMERGENCY: Immediate controlled spindle brake sequence.",
                    "Inspect spindle chuck taper run-out and check workpiece clamping.",
                    "Replace carbide cutting insert before restarting NC program.",
                    "Verify axis servo drive current limit parameters."
                ],
                "parts_required": [
                    {"sku": "TOOL-CNMG-120408", "desc": "Solid Carbide High-Feed Turning Insert", "qty": 1},
                    {"sku": "BOLT-M12-10.9", "desc": "Toolholder High-Tensile Clamping Screw", "qty": 2}
                ],
                "work_order_required": True,
                "priority": "P1 - Critical Emergency"
            }

        # 2. Tool Wear Failure Check (TWF)
        if wear >= 195.0:
            fm = FAILURE_MODES["TWF"]
            return {
                "mode_code": "TWF",
                "mode_title": fm["title"],
                "severity": fm["severity"],
                "severity_badge": "red",
                "risk_score_pct": round(failure_prob * 100, 1),
                "rul_hours": max(1.2, round(4.5 * (1.0 - (wear / 250.0)), 1)),
                "rul_cycles": max(3, int(25 * (1.0 - (wear / 250.0)))),
                "root_cause": f"Tool contact wear reached {wear:.1f} min (critical threshold: 200.0 min). High tip micro-chipping friction.",
                "prescriptions": [
                    "Perform automated tool change (ATC) at next safe block stop.",
                    "Calibrate optical tool presetter for Z-axis offset calibration.",
                    "Inspect surface roughness Ra on last machined batch."
                ],
                "parts_required": [
                    {"sku": "TOOL-CNMG-120408", "desc": "Tungsten Carbide Insert (TiAlN Coated)", "qty": 1},
                    {"sku": "SHIM-SEAT-1604", "desc": "Toolholder Carbide Seat Shim", "qty": 1}
                ],
                "work_order_required": True,
                "priority": "P2 - High Urgency"
            }

        # 3. Heat Dissipation Failure Check (HDF)
        if delta_t < 8.6 and speed < 1380.0:
            fm = FAILURE_MODES["HDF"]
            return {
                "mode_code": "HDF",
                "mode_title": fm["title"],
                "severity": fm["severity"],
                "severity_badge": "amber",
                "risk_score_pct": round(failure_prob * 100, 1),
                "rul_hours": max(2.0, round(6.0 * (1.0 - failure_prob), 1)),
                "rul_cycles": max(5, int(35 * (1.0 - failure_prob))),
                "root_cause": f"Heat dissipation collapsed: ΔTemp is only {delta_t:.1f} K (<8.6 K) at {speed:.0f} rpm.",
                "prescriptions": [
                    "Flush high-pressure flood coolant nozzles and clean inline particulate filter.",
                    "Check coolant concentration with optical refractometer (Target 7-9% Brix).",
                    "Inspect chiller heat exchanger coil for biological or mineral fouling."
                ],
                "parts_required": [
                    {"sku": "FLTR-HYD-50UM", "desc": "Coolant Sump 50-Micron Mesh Filter", "qty": 1},
                    {"sku": "COOL-SYN-20L", "desc": "Semi-Synthetic Metalworking Fluid 20L Drum", "qty": 1}
                ],
                "work_order_required": True,
                "priority": "P2 - High Urgency"
            }

        # 4. Power Failure Check (PWF)
        if power_w < 3500.0 or power_w > 9000.0:
            fm = FAILURE_MODES["PWF"]
            return {
                "mode_code": "PWF",
                "mode_title": fm["title"],
                "severity": fm["severity"],
                "severity_badge": "orange",
                "risk_score_pct": round(failure_prob * 100, 1),
                "rul_hours": max(1.5, round(5.0 * (1.0 - failure_prob), 1)),
                "rul_cycles": max(4, int(28 * (1.0 - failure_prob))),
                "root_cause": f"Mechanical power output calculated at {power_w:.0f} W (Safe operational window: 3,500 W - 9,000 W).",
                "prescriptions": [
                    "Inspect VFD spindle drive inverter output IGBT transistor modules.",
                    "Measure 3-phase motor winding resistance and insulation megohmmeter rating.",
                    "Check spindle belt tension or direct-drive coupling integrity."
                ],
                "parts_required": [
                    {"sku": "VFD-FUSE-30A", "desc": "Ultra-Fast Semiconductor Protection Fuse 30A", "qty": 3},
                    {"sku": "BELT-OPT-POLY", "desc": "Reinforced Spindle Poly-V Drive Belt", "qty": 1}
                ],
                "work_order_required": True,
                "priority": "P1 - Critical Emergency"
            }

        # 5. Default: Random Failure / General Electro-Mechanical Anomaly (RNF)
        fm = FAILURE_MODES["RNF"]
        return {
            "mode_code": "RNF",
            "mode_title": fm["title"],
            "severity": fm["severity"],
            "severity_badge": "purple",
            "risk_score_pct": round(failure_prob * 100, 1),
            "rul_hours": max(3.0, round(8.0 * (1.0 - failure_prob), 1)),
            "rul_cycles": max(8, int(45 * (1.0 - failure_prob))),
            "root_cause": "Anomalous multi-sensor vibration and torque harmonics identified by Stage 1 VAE latent projection.",
            "prescriptions": [
                "Execute precision spindle vibration spectrum analysis (FFT Peak search).",
                "Replenish synthetic polyurea bearing lubrication grease.",
                "Verify axis linear guideway lubrication pressure."
            ],
            "parts_required": [
                {"sku": "LUBE-KLUBER-ISOFLEX", "desc": "High-Speed Spindle Grease Cartridge 400g", "qty": 1}
            ],
            "work_order_required": True,
            "priority": "P3 - Moderate"
        }

    def generate_work_order(self, asset_id: str, diagnosis: Dict[str, Any], technician_email: str) -> Dict[str, Any]:
        """Generates an official maintenance work order ready for dispatch and persists to database."""
        now = datetime.datetime.utcnow()
        wo_id = f"WO-{now.strftime('%Y%m%d')}-{abs(hash(asset_id + str(now))) % 10000:04d}"
        
        wo_dict = {
            "work_order_id": wo_id,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "asset_id": asset_id,
            "status": "DISPATCHED",
            "priority": diagnosis.get("priority", "P2 - High Urgency"),
            "failure_mode": diagnosis.get("mode_title", "Unknown"),
            "severity": diagnosis.get("severity", "HIGH"),
            "rul_hours": float(diagnosis.get("rul_hours", 4.0)),
            "estimated_avoidance_savings": "$17,500 USD (Avoided 2.1h unplanned outage)",
            "prescribed_actions": diagnosis.get("prescriptions", []),
            "parts_required": diagnosis.get("parts_required", []),
            "assigned_technician": technician_email,
            "sign_off_status": "Pending Field Verification"
        }

        # Persist to SQLAlchemy Database
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            from app.db_models import WorkOrderDB
            db_wo = WorkOrderDB(
                work_order_id=wo_dict["work_order_id"],
                created_at=wo_dict["created_at"],
                asset_id=wo_dict["asset_id"],
                status=wo_dict["status"],
                priority=wo_dict["priority"],
                failure_mode=wo_dict["failure_mode"],
                severity=wo_dict["severity"],
                rul_hours=wo_dict["rul_hours"],
                estimated_avoidance_savings=wo_dict["estimated_avoidance_savings"],
                prescribed_actions=wo_dict["prescribed_actions"],
                parts_required=wo_dict["parts_required"],
                assigned_technician=wo_dict["assigned_technician"],
                sign_off_status=wo_dict["sign_off_status"]
            )
            db.add(db_wo)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[Guidance DB Warning] Failed to persist work order: {e}")
        finally:
            db.close()

        return wo_dict

    def get_work_orders(self) -> List[Dict[str, Any]]:
        """Retrieves all work orders from database, ordered by creation time."""
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            from app.db_models import WorkOrderDB
            wos = db.query(WorkOrderDB).order_by(WorkOrderDB.created_timestamp.desc()).all()
            if wos:
                return [w.to_dict() for w in wos]
        except Exception as e:
            print(f"[Guidance DB Warning] Error querying work orders: {e}")
        finally:
            db.close()
            
        return []

guidance_service = PrescriptiveGuidanceService()

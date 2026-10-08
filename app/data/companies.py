# Enterprise Multi-Company & Multi-Industry Domain Profiles
# Demonstrates how the Two-Stage Cross-Domain Deep Learning Architecture
# adapts across heterogeneous industrial companies, machinery types, and cost structures.

ENTERPRISE_COMPANY_PROFILES = {
    "apex_machining": {
        "id": "apex_machining",
        "name": "Apex Precision Machining Corp",
        "industry": "Precision CNC & Aerospace Manufacturing",
        "icon": "fa-gears",
        "theme_color": "blue",
        "facility": "Plant #4 - Stuttgart Advanced Machining Facility",
        "equipment_type": "5-Axis High-Speed CNC Milling Centers & Lathes",
        "fleet_size": 24,
        "operating_mode": "Discrete Batch Manufacturing (High Precision)",
        "downtime_cost_per_hour": 8500,
        "mean_repair_hours": 2.5,
        "false_positive_inspection_cost": 220,
        "optimal_threshold": 0.38,
        "description": "High-precision titanium and Inconel aerospace component milling. High spindle speeds (up to 24,000 RPM) subject cutting inserts and ceramic bearings to intense thermo-mechanical friction.",
        "assets": ["CNC-SPINDLE-402", "CNC-MILL-108", "LATHE-09", "MILL-33", "PUMP-12"],
        "primary_failure_risks": [
            "Tool Flank Wear Failure (TWF)",
            "Spindle Thermal Dissipation Breakdown (HDF)",
            "Electromechanical Power Overload (PWF)",
            "Tool Holder Overstrain (OSF)"
        ],
        "scenarios": {
            "nominal": {
                "name": "Nominal Operating Spindle",
                "badge": "Safe",
                "sensors": {
                    "air_temperature": 298.1,
                    "process_temperature": 308.6,
                    "rotational_speed": 1540.0,
                    "torque": 40.2,
                    "tool_wear": 45.0,
                    "vibration_index": 1.85,
                    "acoustic_emission": 58.2,
                    "oil_pressure": 4.65
                }
            },
            "tool_wear": {
                "name": "Tool Flank Wear (TWF)",
                "badge": "Critical",
                "sensors": {
                    "air_temperature": 299.4,
                    "process_temperature": 310.8,
                    "rotational_speed": 1395.0,
                    "torque": 56.4,
                    "tool_wear": 224.0,
                    "vibration_index": 5.40,
                    "acoustic_emission": 86.5,
                    "oil_pressure": 4.10
                }
            },
            "thermal_choke": {
                "name": "Thermal Dissipation (HDF)",
                "badge": "Warning",
                "sensors": {
                    "air_temperature": 302.5,
                    "process_temperature": 309.2,
                    "rotational_speed": 1240.0,
                    "torque": 64.8,
                    "tool_wear": 142.0,
                    "vibration_index": 4.15,
                    "acoustic_emission": 71.0,
                    "oil_pressure": 3.10
                }
            },
            "overstrain": {
                "name": "Spindle Overstrain (OSF)",
                "badge": "Critical",
                "sensors": {
                    "air_temperature": 299.8,
                    "process_temperature": 311.2,
                    "rotational_speed": 1310.0,
                    "torque": 71.5,
                    "tool_wear": 195.0,
                    "vibration_index": 7.80,
                    "acoustic_emission": 89.2,
                    "oil_pressure": 2.75
                }
            }
        },
        "sensor_schema": [
            {"channel": "air_temperature", "name": "Ambient Temp", "unit": "K", "normal": "298.1 K", "min": 295.0, "max": 305.0, "step": 0.1, "default": 298.1, "role": "Shop floor thermal boundary"},
            {"channel": "process_temperature", "name": "Spindle Temp", "unit": "K", "normal": "308.6 K", "min": 305.0, "max": 315.0, "step": 0.1, "default": 308.6, "role": "Cutting interface heat accumulation"},
            {"channel": "rotational_speed", "name": "Spindle Speed", "unit": "rpm", "normal": "1540 rpm", "min": 1100.0, "max": 2880.0, "step": 10.0, "default": 1540.0, "role": "Kinematic shaft velocity"},
            {"channel": "torque", "name": "Spindle Torque", "unit": "Nm", "normal": "40.2 Nm", "min": 10.0, "max": 80.0, "step": 0.5, "default": 40.2, "role": "Cutting friction drag"},
            {"channel": "tool_wear", "name": "Tool Wear Duration", "unit": "min", "normal": "85 min", "min": 0.0, "max": 250.0, "step": 1.0, "default": 85.0, "role": "Flank wear degradation"},
            {"channel": "vibration_index", "name": "Vibration RMS", "unit": "mm/s", "normal": "2.10 mm/s", "min": 0.5, "max": 10.0, "step": 0.1, "default": 2.1, "role": "High-frequency spindle chatter"},
            {"channel": "acoustic_emission", "name": "Acoustic Noise", "unit": "dB", "normal": "62.0 dB", "min": 45.0, "max": 98.0, "step": 0.5, "default": 62.0, "role": "Sub-surface tool fracture noise"},
            {"channel": "oil_pressure", "name": "Hydraulic Oil Press", "unit": "bar", "normal": "4.50 bar", "min": 2.0, "max": 7.0, "step": 0.1, "default": 4.5, "role": "Spindle lubrication circulation"}
        ],
        "cross_domain_adaptation": {
            "source_transfer_mechanism": "HVAC Chiller thermodynamic entropy accumulation maps directly to CNC Spindle thermal dissipation collapse (Process Temp - Air Temp differential).",
            "alignment_dimension": "8D Latent Projection (W_p in R^{8x8})",
            "transfer_lift": "+15.8% Recall over baseline RF without cross-domain features",
            "compliance_standards": ["ISO 13849-1", "ISO 10816 Mechanical Vibration"]
        }
    },

    "aeolus_wind": {
        "id": "aeolus_wind",
        "name": "Aeolus Offshore Wind Energy",
        "industry": "Renewable Utilities & Clean Power Generation",
        "icon": "fa-wind",
        "theme_color": "cyan",
        "facility": "North Sea Offshore Wind Farm - Cluster Bravo",
        "equipment_type": "3.5 MW Direct-Drive & Geared Wind Turbines",
        "fleet_size": 48,
        "operating_mode": "Continuous Energy Generation (Weather-Modulated)",
        "downtime_cost_per_hour": 32000,
        "mean_repair_hours": 12.0,
        "false_positive_inspection_cost": 3500,
        "optimal_threshold": 0.28,
        "description": "Offshore wind turbines operating in extreme marine weather. Mechanical failures require specialized jack-up vessel mobilization costing up to $200,000 per voyage. Early warning of bearing micropitting is mission-critical.",
        "assets": ["WTG-OFFSHORE-07", "WTG-OFFSHORE-12", "WTG-DEEPSEA-03", "WTG-TURBINE-22"],
        "primary_failure_risks": [
            "Epicyclic Gearbox Bearing Micropitting (GBF)",
            "Main Shaft Lubrication Starvation (MLF)",
            "Pitch Actuator Hydraulic Depressurization (PIF)",
            "Generator Stator Winding Insulation Breakdown (GWF)"
        ],
        "scenarios": {
            "nominal": {
                "name": "Nominal 3.5MW Turbine",
                "badge": "Safe",
                "sensors": {
                    "ambient_marine_temp": 12.4,
                    "gearbox_oil_temp": 68.2,
                    "rotor_rpm": 14.2,
                    "generator_torque": 1820.0,
                    "main_bearing_vibe": 1.85,
                    "high_speed_vibe": 3.40,
                    "hydraulic_pitch_press": 180.0,
                    "generator_phase_current": 690.0
                }
            },
            "micropitting": {
                "name": "Gearbox Micropitting (GBF)",
                "badge": "Critical",
                "sensors": {
                    "ambient_marine_temp": 14.8,
                    "gearbox_oil_temp": 84.5,
                    "rotor_rpm": 15.6,
                    "generator_torque": 2050.0,
                    "main_bearing_vibe": 2.80,
                    "high_speed_vibe": 6.20,
                    "hydraulic_pitch_press": 175.0,
                    "generator_phase_current": 740.0
                }
            },
            "pitch_loss": {
                "name": "Pitch Depressurization (PIF)",
                "badge": "Warning",
                "sensors": {
                    "ambient_marine_temp": 9.5,
                    "gearbox_oil_temp": 71.0,
                    "rotor_rpm": 11.2,
                    "generator_torque": 1450.0,
                    "main_bearing_vibe": 2.10,
                    "high_speed_vibe": 3.80,
                    "hydraulic_pitch_press": 138.0,
                    "generator_phase_current": 580.0
                }
            },
            "shaft_fatigue": {
                "name": "Main Shaft Bearing Fatigue",
                "badge": "Critical",
                "sensors": {
                    "ambient_marine_temp": 11.2,
                    "gearbox_oil_temp": 74.0,
                    "rotor_rpm": 13.8,
                    "generator_torque": 1890.0,
                    "main_bearing_vibe": 4.40,
                    "high_speed_vibe": 4.20,
                    "hydraulic_pitch_press": 178.0,
                    "generator_phase_current": 710.0
                }
            }
        },
        "sensor_schema": [
            {"channel": "ambient_marine_temp", "name": "Sea Ambient Temp", "unit": "°C", "normal": "12.4 °C", "min": -5.0, "max": 35.0, "step": 0.5, "default": 12.4, "role": "Nacelle boundary conditions"},
            {"channel": "gearbox_oil_temp", "name": "Gearbox Sump Temp", "unit": "°C", "normal": "68.2 °C", "min": 40.0, "max": 95.0, "step": 0.5, "default": 68.2, "role": "Bulk lubricating oil thermal state"},
            {"channel": "rotor_rpm", "name": "Main Rotor Speed", "unit": "rpm", "normal": "14.2 rpm", "min": 5.0, "max": 25.0, "step": 0.1, "default": 14.2, "role": "Aerodynamic blade rotation"},
            {"channel": "generator_torque", "name": "Electromagnetic Torque", "unit": "kNm", "normal": "1820 kNm", "min": 500.0, "max": 2500.0, "step": 10.0, "default": 1820.0, "role": "Grid load resistance"},
            {"channel": "main_bearing_vibe", "name": "Main Bearing Vibe", "unit": "mm/s", "normal": "1.85 mm/s", "min": 0.5, "max": 6.0, "step": 0.05, "default": 1.85, "role": "Low-speed shaft radial deflection"},
            {"channel": "high_speed_vibe", "name": "High-Speed Pinion Vibe", "unit": "mm/s", "normal": "3.40 mm/s", "min": 1.0, "max": 8.0, "step": 0.05, "default": 3.40, "role": "Planetary gear mesh harmonic"},
            {"channel": "hydraulic_pitch_press", "name": "Pitch Cylinder Press", "unit": "bar", "normal": "180 bar", "min": 120.0, "max": 220.0, "step": 1.0, "default": 180.0, "role": "Aerodynamic blade pitch control"},
            {"channel": "generator_phase_current", "name": "Stator Current RMS", "unit": "A", "normal": "690 A", "min": 200.0, "max": 900.0, "step": 5.0, "default": 690.0, "role": "Electrical power generation load"}
        ],
        "cross_domain_adaptation": {
            "source_transfer_mechanism": "HVAC compressor vibration harmonics and fan cooling air flow map into turbine nacelle ventilation and planetary gearbox bearing degradation.",
            "alignment_dimension": "8D Latent Projection (W_p in R^{8x8})",
            "transfer_lift": "+18.2% Early detection window (48 hours earlier warning)",
            "compliance_standards": ["IEC 61400-25 Wind Turbines", "ISO 14224 Reliability Data"]
        }
    },

    "petroflow_refining": {
        "id": "petroflow_refining",
        "name": "PetroFlow Refining & Chemical Process",
        "industry": "Petrochemical Refining & Continuous Fluid Process",
        "icon": "fa-flask-vial",
        "theme_color": "amber",
        "facility": "Rotterdam Hydrocracker Unit #2",
        "equipment_type": "Multi-Stage API 610 Centrifugal Slurry Pumps & Gas Compressors",
        "fleet_size": 36,
        "operating_mode": "24/7 Continuous Fluid Flow (Zero Tolerance for Shutdown)",
        "downtime_cost_per_hour": 65000,
        "mean_repair_hours": 6.0,
        "false_positive_inspection_cost": 850,
        "optimal_threshold": 0.22,
        "description": "Critical hydrocracker feed pumps circulating corrosive hydrocarbons at 350°C and 140 bar. Mechanical seal leaks pose immediate fire and catastrophic flaring hazards requiring extreme fail-safe vigilance.",
        "assets": ["PUMP-API610-P201A", "SLURRY-PUMP-P104", "CRACKER-FEED-P302", "GAS-COMP-K101"],
        "primary_failure_risks": [
            "Mechanical Barrier Seal Flush Failure (MSF)",
            "Impeller Cavitation Erosion & Vapor Collapse (CEF)",
            "Thrust Bearing Hydrodynamic Seizure (TBF)",
            "Casing Thermal Stress Crack (TCF)"
        ],
        "scenarios": {
            "nominal": {
                "name": "Nominal API-610 Pump",
                "badge": "Safe",
                "sensors": {
                    "suction_temp": 285.0,
                    "casing_skin_temp": 312.4,
                    "pump_shaft_speed": 2980.0,
                    "discharge_pressure": 135.0,
                    "differential_pressure": 95.5,
                    "seal_flush_flow": 18.5,
                    "casing_cavitation_noise": 54.2,
                    "thrust_bearing_vibe": 1.45
                }
            },
            "cavitation": {
                "name": "Impeller Cavitation (CEF)",
                "badge": "Critical",
                "sensors": {
                    "suction_temp": 292.0,
                    "casing_skin_temp": 320.0,
                    "pump_shaft_speed": 2990.0,
                    "discharge_pressure": 118.0,
                    "differential_pressure": 71.0,
                    "seal_flush_flow": 17.0,
                    "casing_cavitation_noise": 78.5,
                    "thrust_bearing_vibe": 2.85
                }
            },
            "seal_starve": {
                "name": "Barrier Seal Starvation (MSF)",
                "badge": "Warning",
                "sensors": {
                    "suction_temp": 288.0,
                    "casing_skin_temp": 346.0,
                    "pump_shaft_speed": 2980.0,
                    "discharge_pressure": 134.0,
                    "differential_pressure": 94.0,
                    "seal_flush_flow": 9.2,
                    "casing_cavitation_noise": 58.0,
                    "thrust_bearing_vibe": 2.10
                }
            },
            "thrust_seize": {
                "name": "Thrust Bearing Hydrodynamic Seizure",
                "badge": "Critical",
                "sensors": {
                    "suction_temp": 286.0,
                    "casing_skin_temp": 328.0,
                    "pump_shaft_speed": 2960.0,
                    "discharge_pressure": 132.0,
                    "differential_pressure": 92.0,
                    "seal_flush_flow": 16.5,
                    "casing_cavitation_noise": 62.0,
                    "thrust_bearing_vibe": 3.85
                }
            }
        },
        "sensor_schema": [
            {"channel": "suction_temp", "name": "Process Inlet Temp", "unit": "°C", "normal": "285.0 °C", "min": 240.0, "max": 330.0, "step": 1.0, "default": 285.0, "role": "Hydrocarbon suction temperature"},
            {"channel": "casing_skin_temp", "name": "Pump Casing Temp", "unit": "°C", "normal": "312.4 °C", "min": 260.0, "max": 360.0, "step": 1.0, "default": 312.4, "role": "Thermal boundary gradient"},
            {"channel": "pump_shaft_speed", "name": "Impeller Shaft Speed", "unit": "rpm", "normal": "2980 rpm", "min": 2200.0, "max": 3500.0, "step": 10.0, "default": 2980.0, "role": "High-velocity fluid displacement"},
            {"channel": "discharge_pressure", "name": "Discharge Head Press", "unit": "bar", "normal": "135.0 bar", "min": 90.0, "max": 160.0, "step": 0.5, "default": 135.0, "role": "Process delivery backpressure"},
            {"channel": "differential_pressure", "name": "Head Differential", "unit": "bar", "normal": "95.5 bar", "min": 60.0, "max": 120.0, "step": 0.5, "default": 95.5, "role": "Effective hydraulic head gain"},
            {"channel": "seal_flush_flow", "name": "API Plan 53B Flush Flow", "unit": "L/min", "normal": "18.5 L/min", "min": 5.0, "max": 25.0, "step": 0.2, "default": 18.5, "role": "Barrier seal cooling circulation"},
            {"channel": "casing_cavitation_noise", "name": "Cavitation Acoustic Noise", "unit": "dB", "normal": "54.2 dB", "min": 40.0, "max": 88.0, "step": 0.5, "default": 54.2, "role": "High-frequency bubble collapse"},
            {"channel": "thrust_bearing_vibe", "name": "Axial Bearing Vibe", "unit": "mm/s", "normal": "1.45 mm/s", "min": 0.5, "max": 5.0, "step": 0.05, "default": 1.45, "role": "Hydraulic axial thrust balance"}
        ],
        "cross_domain_adaptation": {
            "source_transfer_mechanism": "HVAC chilled water return/supply delta and flow static pressure dynamics transfer directly to closed-loop chemical heat exchangers and pump cavitation mechanics.",
            "alignment_dimension": "8D Latent Projection (W_p in R^{8x8})",
            "transfer_lift": "Zero Catastrophic Hydrocarbon Leaks (100% Fail-Safe Seal Recall)",
            "compliance_standards": ["API 610 / API 682 Mechanical Seals", "IEC 61508 SIL-3"]
        }
    },

    "biofreeze_pharma": {
        "id": "biofreeze_pharma",
        "name": "BioFreeze Cryogenics & Biologics",
        "industry": "Sterile Pharmaceutical Cold Chain & Freeze-Drying",
        "icon": "fa-snowflake",
        "theme_color": "emerald",
        "facility": "Basel Biologics Formulation Center - Suite C",
        "equipment_type": "Cascade Refrigeration (-80°C) Compressors & Industrial Lyophilizers",
        "fleet_size": 18,
        "operating_mode": "Ultra-Critical Sterile Batch Lyophilization (72-hour Cycles)",
        "downtime_cost_per_hour": 110000,
        "mean_repair_hours": 4.0,
        "false_positive_inspection_cost": 450,
        "optimal_threshold": 0.18,
        "description": "Sterile lyophilization of high-value oncology antibodies and mRNA vaccines. A 20-minute chiller malfunction ruins a $3,500,000 biological batch. The architecture requires absolute zero false-negative guarantees.",
        "assets": ["CRYO-LYO-04", "CASCADE-CHILLER-01", "FREEZE-DRYER-08", "VACCINE-VAULT-02"],
        "primary_failure_risks": [
            "Low-Stage Compressor Suction Flooding (CSF)",
            "Thermal Shelf Vacuum Desorption Collapse (SVC)",
            "Expansion Valve Cryogenic Orifice Freeze-Up (EOF)",
            "Condenser Defrost Thermal Runaway (CTR)"
        ],
        "scenarios": {
            "nominal": {
                "name": "Nominal -80°C Cryo Lyo",
                "badge": "Safe",
                "sensors": {
                    "shelf_product_temp": -48.5,
                    "condenser_surface_temp": -78.2,
                    "cascade_low_stage_rpm": 1750.0,
                    "suction_vapor_press": 0.12,
                    "discharge_superheat": 18.4,
                    "oil_separator_temp": 55.0,
                    "compressor_vibration": 1.10,
                    "expansion_valve_opening": 240.0
                }
            },
            "condenser_choke": {
                "name": "Vacuum Desorption Collapse (SVC)",
                "badge": "Critical",
                "sensors": {
                    "shelf_product_temp": -41.0,
                    "condenser_surface_temp": -62.0,
                    "cascade_low_stage_rpm": 1820.0,
                    "suction_vapor_press": 0.28,
                    "discharge_superheat": 12.0,
                    "oil_separator_temp": 58.0,
                    "compressor_vibration": 1.50,
                    "expansion_valve_opening": 290.0
                }
            },
            "liquid_flood": {
                "name": "Compressor Suction Flooding (CSF)",
                "badge": "Critical",
                "sensors": {
                    "shelf_product_temp": -47.0,
                    "condenser_surface_temp": -74.0,
                    "cascade_low_stage_rpm": 1740.0,
                    "suction_vapor_press": 0.15,
                    "discharge_superheat": 5.2,
                    "oil_separator_temp": 42.0,
                    "compressor_vibration": 2.40,
                    "expansion_valve_opening": 360.0
                }
            },
            "eev_freeze": {
                "name": "Expansion Valve Orifice Freeze (EOF)",
                "badge": "Warning",
                "sensors": {
                    "shelf_product_temp": -44.0,
                    "condenser_surface_temp": -70.0,
                    "cascade_low_stage_rpm": 1780.0,
                    "suction_vapor_press": 0.08,
                    "discharge_superheat": 24.0,
                    "oil_separator_temp": 63.0,
                    "compressor_vibration": 1.80,
                    "expansion_valve_opening": 395.0
                }
            }
        },
        "sensor_schema": [
            {"channel": "shelf_product_temp", "name": "Core Product Temp", "unit": "°C", "normal": "-48.5 °C", "min": -60.0, "max": -20.0, "step": 0.5, "default": -48.5, "role": "Freeze-drying product sublimating cake"},
            {"channel": "condenser_surface_temp", "name": "Ice Condenser Temp", "unit": "°C", "normal": "-78.2 °C", "min": -90.0, "max": -50.0, "step": 0.5, "default": -78.2, "role": "Ultra-low ice trapping surface"},
            {"channel": "cascade_low_stage_rpm", "name": "Low-Stage Compressor", "unit": "rpm", "normal": "1750 rpm", "min": 1200.0, "max": 2400.0, "step": 10.0, "default": 1750.0, "role": "R-23 cryogenic vapor compressor"},
            {"channel": "suction_vapor_press", "name": "Suction Pressure", "unit": "mbar", "normal": "0.12 mbar", "min": 0.02, "max": 0.50, "step": 0.01, "default": 0.12, "role": "Sublimation chamber vacuum head"},
            {"channel": "discharge_superheat", "name": "Discharge Superheat", "unit": "K", "normal": "18.4 K", "min": 2.0, "max": 35.0, "step": 0.2, "default": 18.4, "role": "Refrigerant thermodynamic state"},
            {"channel": "oil_separator_temp", "name": "Oil Sump Temp", "unit": "°C", "normal": "55.0 °C", "min": 30.0, "max": 75.0, "step": 0.5, "default": 55.0, "role": "Synthetic polyolester lube state"},
            {"channel": "compressor_vibration", "name": "Scroll Vibration RMS", "unit": "mm/s", "normal": "1.10 mm/s", "min": 0.4, "max": 4.0, "step": 0.05, "default": 1.10, "role": "Hermetic scroll mechanical wear"},
            {"channel": "expansion_valve_opening", "name": "Electronic EEV Step", "unit": "steps", "normal": "240 steps", "min": 100.0, "max": 450.0, "step": 5.0, "default": 240.0, "role": "Cryogenic liquid mass modulation"}
        ],
        "cross_domain_adaptation": {
            "source_transfer_mechanism": "Direct 1:1 thermodynamic domain isomorphism: Source HVAC enthalpy, chilled water loops, and compressor superheat map natively with near-zero adaptation loss to Cryogenic Lyophilizers.",
            "alignment_dimension": "8D Latent Projection (W_p in R^{8x8})",
            "transfer_lift": "Zero Spoiled Biologic Batches ($14,000,000+ batch inventory saved annually)",
            "compliance_standards": ["FDA 21 CFR Part 11", "EU GMP Annex 1 Sterile Manufacturing"]
        }
    }
}

def get_all_companies():
    """Retrieves all company profiles from SQLAlchemy database, falling back to in-memory profiles."""
    try:
        from app.database import SessionLocal
        from app.db_models import CompanyDB
        db = SessionLocal()
        try:
            db_comps = db.query(CompanyDB).all()
            if db_comps:
                merged = dict(ENTERPRISE_COMPANY_PROFILES)
                for c in db_comps:
                    merged[c.id] = c.to_dict()
                return merged
        finally:
            db.close()
    except Exception as e:
        print(f"[Companies DB Warning] Error querying companies: {e}")

    return ENTERPRISE_COMPANY_PROFILES

def save_company_profile(comp_dict: dict):
    """Persists a new or updated company profile into SQLAlchemy database."""
    ENTERPRISE_COMPANY_PROFILES[comp_dict["id"]] = comp_dict
    try:
        from app.database import SessionLocal
        from app.db_models import CompanyDB
        db = SessionLocal()
        try:
            existing = db.query(CompanyDB).filter(CompanyDB.id == comp_dict["id"]).first()
            if existing:
                for k, v in comp_dict.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
            else:
                comp_obj = CompanyDB(
                    id=comp_dict["id"],
                    name=comp_dict["name"],
                    industry=comp_dict["industry"],
                    icon=comp_dict.get("icon", "building-2"),
                    theme_color=comp_dict.get("theme_color", "indigo"),
                    facility=comp_dict.get("facility", ""),
                    equipment_type=comp_dict.get("equipment_type", ""),
                    fleet_size=comp_dict.get("fleet_size", 12),
                    operating_mode=comp_dict.get("operating_mode", ""),
                    downtime_cost_per_hour=float(comp_dict.get("downtime_cost_per_hour", 8500.0)),
                    mean_repair_hours=float(comp_dict.get("mean_repair_hours", 2.5)),
                    false_positive_inspection_cost=float(comp_dict.get("false_positive_inspection_cost", 350.0)),
                    optimal_threshold=float(comp_dict.get("optimal_threshold", 0.38)),
                    description=comp_dict.get("description", ""),
                    assets=comp_dict.get("assets", []),
                    primary_failure_risks=comp_dict.get("primary_failure_risks", []),
                    scenarios=comp_dict.get("scenarios", {}),
                    sensor_schema=comp_dict.get("sensor_schema", []),
                    cross_domain_adaptation=comp_dict.get("cross_domain_adaptation", {})
                )
                db.add(comp_obj)
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[Companies DB Warning] Error saving company {comp_dict.get('id')}: {e}")


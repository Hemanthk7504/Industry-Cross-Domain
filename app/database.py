import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.config import settings

# -------------------------------------------------------------
# Database Engine Configuration (SQLite / PostgreSQL)
# -------------------------------------------------------------
# Default: sqlite:///./industrial_cross_domain.db
# PostgreSQL (Later): postgresql+psycopg2://user:password@localhost:5432/crossdomain_db
db_url = settings.DATABASE_URL

# Fix for Heroku/Render legacy postgres:// URI if used
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url.startswith("postgresql://") and "+psycopg2" not in db_url and "+asyncpg" not in db_url:
    # Ensure standard psycopg2 dialect is used if available
    pass

is_sqlite = db_url.startswith("sqlite")

if is_sqlite:
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    # Production PostgreSQL pool configuration
    engine = create_engine(
        db_url,
        pool_size=15,
        max_overflow=25,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables and seeds initial industrial records."""
    from app.db_models import UserDB, CompanyDB, AuditLogDB, WorkOrderDB, PredictionDB
    from app.auth import USERS_DB
    from app.data.companies import ENTERPRISE_COMPANY_PROFILES

    # Create all tables if they don't exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 1. Seed Users if table is empty
        user_count = db.query(UserDB).count()
        if user_count == 0:
            for email, u in USERS_DB.items():
                user_obj = UserDB(
                    email=u["email"],
                    name=u["name"],
                    role=u.get("role", "engineer"),
                    role_title=u.get("role_title", "Senior Maintenance Engineer"),
                    department=u.get("department", "Operations"),
                    avatar=u.get("avatar", "US"),
                    password_hash=u["password_hash"]
                )
                db.add(user_obj)
            db.commit()

        # 2. Seed Enterprise Companies if table is empty
        company_count = db.query(CompanyDB).count()
        if company_count == 0:
            for cid, c in ENTERPRISE_COMPANY_PROFILES.items():
                comp_obj = CompanyDB(
                    id=c["id"],
                    name=c["name"],
                    industry=c["industry"],
                    icon=c.get("icon", "building-2"),
                    theme_color=c.get("theme_color", "indigo"),
                    facility=c.get("facility", ""),
                    equipment_type=c.get("equipment_type", ""),
                    fleet_size=c.get("fleet_size", 12),
                    operating_mode=c.get("operating_mode", ""),
                    downtime_cost_per_hour=float(c.get("downtime_cost_per_hour", 8500.0)),
                    mean_repair_hours=float(c.get("mean_repair_hours", 2.5)),
                    false_positive_inspection_cost=float(c.get("false_positive_inspection_cost", 350.0)),
                    optimal_threshold=float(c.get("optimal_threshold", 0.38)),
                    description=c.get("description", ""),
                    assets=c.get("assets", []),
                    primary_failure_risks=c.get("primary_failure_risks", []),
                    scenarios=c.get("scenarios", {}),
                    sensor_schema=c.get("sensor_schema", []),
                    cross_domain_adaptation=c.get("cross_domain_adaptation", {})
                )
                db.add(comp_obj)
            db.commit()

        # 3. Seed Audit Logs if table is empty
        audit_count = db.query(AuditLogDB).count()
        if audit_count == 0:
            default_audits = [
                {
                    "id": "AUD-2026-9041",
                    "timestamp": "2026-10-08 11:42:15 UTC",
                    "asset_id": "CNC-SPINDLE-402",
                    "operator_email": "engineer@plant.com",
                    "operator_role": "Senior Maintenance Engineer",
                    "model_name": "BiLSTM-BiGRU-VAE v2.4",
                    "prediction": "Normal Operational State",
                    "failure_probability": 0.042,
                    "confidence": 0.992,
                    "diagnosis_code": "NOMINAL",
                    "action_taken": "Routine Operation Cleared",
                    "hash": "e8f3b218a09bc345d12ef90123456789",
                    "governance_status": "APPROVED",
                    "drift_ks_stat": 0.021
                },
                {
                    "id": "AUD-2026-9038",
                    "timestamp": "2026-10-08 10:15:33 UTC",
                    "asset_id": "CHILLER-AHU-03",
                    "operator_email": "lead@plant.com",
                    "operator_role": "Lead Reliability Specialist",
                    "model_name": "BiLSTM-BiGRU-VAE v2.4",
                    "prediction": "Imminent Failure Detected",
                    "failure_probability": 0.982,
                    "confidence": 0.988,
                    "diagnosis_code": "TWF",
                    "action_taken": "Work Order WO-20261008-0108 Dispatched",
                    "hash": "7a9c84e1b305f284c0128da1b98234ea",
                    "governance_status": "ACTIONED",
                    "drift_ks_stat": 0.034
                },
                {
                    "id": "AUD-2026-9031",
                    "timestamp": "2026-10-08 08:50:11 UTC",
                    "asset_id": "MILL-AXIS-33",
                    "operator_email": "engineer@plant.com",
                    "operator_role": "Senior Maintenance Engineer",
                    "model_name": "BiLSTM-BiGRU-VAE v2.4",
                    "prediction": "Imminent Failure Detected",
                    "failure_probability": 0.995,
                    "confidence": 0.997,
                    "diagnosis_code": "OSF",
                    "action_taken": "Emergency Braking Executed",
                    "hash": "4f1a23e980cb6144e5904bc3827104ae",
                    "governance_status": "ACTIONED",
                    "drift_ks_stat": 0.029
                }
            ]
            for a in default_audits:
                audit_obj = AuditLogDB(
                    id=a["id"],
                    timestamp=a["timestamp"],
                    asset_id=a["asset_id"],
                    operator_email=a["operator_email"],
                    operator_role=a["operator_role"],
                    model_name=a["model_name"],
                    prediction=a["prediction"],
                    failure_probability=float(a["failure_probability"]),
                    confidence=float(a["confidence"]),
                    diagnosis_code=a["diagnosis_code"],
                    action_taken=a["action_taken"],
                    hash=a["hash"],
                    governance_status=a["governance_status"],
                    drift_ks_stat=float(a["drift_ks_stat"])
                )
                db.add(audit_obj)
            db.commit()

        # 4. Seed Initial Work Orders if table is empty
        wo_count = db.query(WorkOrderDB).count()
        if wo_count == 0:
            default_wos = [
                {
                    "work_order_id": "WO-20261008-9031",
                    "created_at": "Today, 08:50 UTC",
                    "asset_id": "MILL-33 (Bed Milling Machine)",
                    "status": "DISPATCHED",
                    "priority": "P1 - Critical Emergency",
                    "failure_mode": "OSF - Overstrain Failure",
                    "severity": "CRITICAL",
                    "rul_hours": 0.8,
                    "estimated_avoidance_savings": "$21,250 USD",
                    "prescribed_actions": [
                        "EMERGENCY: Immediate controlled spindle brake sequence.",
                        "Inspect spindle chuck taper run-out and replace carbide cutting insert."
                    ],
                    "parts_required": [
                        {"sku": "TOOL-CNMG-120408", "desc": "Solid Carbide Insert", "qty": 1},
                        {"sku": "BOLT-M12-10.9", "desc": "Toolholder High-Tensile Screw", "qty": 2}
                    ],
                    "assigned_technician": "Marcus Vance (Sr. Engineer)",
                    "sign_off_status": "Pending Field Verification"
                },
                {
                    "work_order_id": "WO-20261008-9024",
                    "created_at": "Today, 07:12 UTC",
                    "asset_id": "CNC-MILL-108 (5-Axis Center)",
                    "status": "DISPATCHED",
                    "priority": "P2 - High Urgency",
                    "failure_mode": "TWF - Tool Flank Wear Failure",
                    "severity": "HIGH",
                    "rul_hours": 3.4,
                    "estimated_avoidance_savings": "$18,500 USD",
                    "prescribed_actions": [
                        "Perform automated tool change (ATC) at next safe block stop.",
                        "Calibrate optical tool presetter for Z-axis offset calibration."
                    ],
                    "parts_required": [
                        {"sku": "TOOL-CNMG-120408", "desc": "Tungsten Carbide Insert (TiAlN)", "qty": 1},
                        {"sku": "SHIM-SEAT-1604", "desc": "Toolholder Carbide Seat Shim", "qty": 1}
                    ],
                    "assigned_technician": "Elena Rostova (Lead Specialist)",
                    "sign_off_status": "Scheduled for Shift Change"
                },
                {
                    "work_order_id": "WO-20261008-8995",
                    "created_at": "Yesterday, 22:40 UTC",
                    "asset_id": "CHILLER-AHU-03 (Refrigeration)",
                    "status": "IN_PROGRESS",
                    "priority": "P2 - High Urgency",
                    "failure_mode": "HDF - Heat Dissipation Failure",
                    "severity": "HIGH",
                    "rul_hours": 5.2,
                    "estimated_avoidance_savings": "$24,750 USD",
                    "prescribed_actions": [
                        "Clean evaporator intake louvers and flush particulate buildup.",
                        "Check refrigerant sight glass for flash gas bubbling."
                    ],
                    "parts_required": [
                        {"sku": "FILTER-HVAC-M13", "desc": "High-Efficiency Air Filter Panel", "qty": 4},
                        {"sku": "R134A-CYL-12KG", "desc": "Refrigerant Cylinder Charge", "qty": 1}
                    ],
                    "assigned_technician": "Arthur Pendelton (Plant Operations)",
                    "sign_off_status": "Work In Progress"
                }
            ]
            for wo in default_wos:
                wo_obj = WorkOrderDB(
                    work_order_id=wo["work_order_id"],
                    created_at=wo["created_at"],
                    asset_id=wo["asset_id"],
                    status=wo["status"],
                    priority=wo["priority"],
                    failure_mode=wo["failure_mode"],
                    severity=wo["severity"],
                    rul_hours=float(wo["rul_hours"]),
                    estimated_avoidance_savings=wo["estimated_avoidance_savings"],
                    prescribed_actions=wo["prescribed_actions"],
                    parts_required=wo["parts_required"],
                    assigned_technician=wo["assigned_technician"],
                    sign_off_status=wo["sign_off_status"]
                )
                db.add(wo_obj)
            db.commit()

    except Exception as e:
        db.rollback()
        print(f"[Database Init Warning] Error during initial seeding: {e}")
    finally:
        db.close()

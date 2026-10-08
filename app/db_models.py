from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, JSON
from app.database import Base

class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(50), default="engineer", nullable=False)
    role_title = Column(String(255), default="Senior Maintenance Engineer")
    department = Column(String(255), default="Industrial Operations")
    avatar = Column(String(10), default="US")
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "role_title": self.role_title,
            "department": self.department,
            "avatar": self.avatar,
            "password_hash": self.password_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class CompanyDB(Base):
    __tablename__ = "companies"

    id = Column(String(100), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    industry = Column(String(255), nullable=False)
    icon = Column(String(50), default="building-2")
    theme_color = Column(String(50), default="indigo")
    facility = Column(String(255), default="")
    equipment_type = Column(String(255), default="")
    fleet_size = Column(Integer, default=12)
    operating_mode = Column(String(255), default="")
    downtime_cost_per_hour = Column(Float, default=8500.0)
    mean_repair_hours = Column(Float, default=2.5)
    false_positive_inspection_cost = Column(Float, default=350.0)
    optimal_threshold = Column(Float, default=0.38)
    description = Column(Text, default="")
    assets = Column(JSON, default=list)
    primary_failure_risks = Column(JSON, default=list)
    scenarios = Column(JSON, default=dict)
    sensor_schema = Column(JSON, default=list)
    cross_domain_adaptation = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "industry": self.industry,
            "icon": self.icon,
            "theme_color": self.theme_color,
            "facility": self.facility,
            "equipment_type": self.equipment_type,
            "fleet_size": self.fleet_size,
            "operating_mode": self.operating_mode,
            "downtime_cost_per_hour": self.downtime_cost_per_hour,
            "mean_repair_hours": self.mean_repair_hours,
            "false_positive_inspection_cost": self.false_positive_inspection_cost,
            "optimal_threshold": self.optimal_threshold,
            "description": self.description,
            "assets": self.assets or [],
            "primary_failure_risks": self.primary_failure_risks or [],
            "scenarios": self.scenarios or {},
            "sensor_schema": self.sensor_schema or [],
            "cross_domain_adaptation": self.cross_domain_adaptation or {}
        }


class AuditLogDB(Base):
    __tablename__ = "audit_logs"

    id = Column(String(100), primary_key=True, index=True)
    timestamp = Column(String(100), nullable=False)
    asset_id = Column(String(100), index=True, nullable=False)
    operator_email = Column(String(255), default="engineer@plant.com")
    operator_role = Column(String(255), default="Senior Maintenance Engineer")
    model_name = Column(String(255), default="BiLSTM-BiGRU-VAE v2.4")
    prediction = Column(String(255), default="Normal Operational State")
    failure_probability = Column(Float, default=0.0)
    confidence = Column(Float, default=0.99)
    diagnosis_code = Column(String(50), default="NOMINAL")
    action_taken = Column(String(255), default="Routine Operation Cleared")
    hash = Column(String(255), default="")
    governance_status = Column(String(50), default="APPROVED")
    drift_ks_stat = Column(Float, default=0.02)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "operator_email": self.operator_email,
            "operator_role": self.operator_role,
            "model_name": self.model_name,
            "prediction": self.prediction,
            "failure_probability": self.failure_probability,
            "confidence": self.confidence,
            "diagnosis_code": self.diagnosis_code,
            "action_taken": self.action_taken,
            "hash": self.hash,
            "governance_status": self.governance_status,
            "drift_ks_stat": self.drift_ks_stat
        }


class WorkOrderDB(Base):
    __tablename__ = "work_orders"

    work_order_id = Column(String(100), primary_key=True, index=True)
    created_at = Column(String(100), nullable=False)
    asset_id = Column(String(100), index=True, nullable=False)
    status = Column(String(50), default="DISPATCHED")
    priority = Column(String(100), default="P2 - High Urgency")
    failure_mode = Column(String(255), default="Unknown")
    severity = Column(String(50), default="HIGH")
    rul_hours = Column(Float, default=4.0)
    estimated_avoidance_savings = Column(String(255), default="$17,500 USD")
    prescribed_actions = Column(JSON, default=list)
    parts_required = Column(JSON, default=list)
    assigned_technician = Column(String(255), default="engineer@plant.com")
    sign_off_status = Column(String(100), default="Pending Field Verification")
    created_timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "work_order_id": self.work_order_id,
            "created_at": self.created_at,
            "asset_id": self.asset_id,
            "status": self.status,
            "priority": self.priority,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "rul_hours": self.rul_hours,
            "estimated_avoidance_savings": self.estimated_avoidance_savings,
            "prescribed_actions": self.prescribed_actions or [],
            "parts_required": self.parts_required or [],
            "assigned_technician": self.assigned_technician,
            "sign_off_status": self.sign_off_status
        }


class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    asset_id = Column(String(100), index=True)
    company_id = Column(String(100), default="apex_machining")
    predicted_failure = Column(Boolean, default=False)
    failure_probability = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.95)
    threshold = Column(Float, default=0.38)
    target_sensors = Column(JSON, default=dict)
    source_hvac_sensors = Column(JSON, default=dict)
    stage1_vae = Column(JSON, default=dict)
    diagnosis = Column(JSON, default=dict)
    maintenance_guidance = Column(JSON, default=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "asset_id": self.asset_id,
            "company_id": self.company_id,
            "predicted_failure": self.predicted_failure,
            "failure_probability": self.failure_probability,
            "confidence_score": self.confidence_score,
            "threshold": self.threshold,
            "target_sensors": self.target_sensors or {},
            "source_hvac_sensors": self.source_hvac_sensors or {},
            "stage1_vae": self.stage1_vae or {},
            "diagnosis": self.diagnosis or {},
            "maintenance_guidance": self.maintenance_guidance or {}
        }

import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Two-Stage Cross-Domain Predictive Maintenance System"
    VERSION: str = "2.4.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secure-cross-domain-industrial-jwt-key-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # -------------------------------------------------------------
    # Database Configuration (SQLAlchemy)
    # -------------------------------------------------------------
    # Default: Local SQLite database
    # To switch to PostgreSQL later, set the DATABASE_URL environment variable:
    # Example: DATABASE_URL="postgresql+psycopg2://username:password@localhost:5432/cross_domain_ai"
    # Example: DATABASE_URL="postgresql://username:password@db.example.com:5432/industrial_db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./industrial_cross_domain.db")
    
    # Industrial Thresholds & Costs
    DEFAULT_THRESHOLD: float = 0.38
    COST_UNPLANNED_DOWNTIME_PER_HOUR: float = 8500.0  # USD
    COST_PREVENTIVE_INSPECTION: float = 450.0  # USD
    COST_FALSE_ALARM_CHECK: float = 220.0  # USD

    # Latent embedding dimensions
    STAGE1_LATENT_DIM: int = 8
    CHAMPION_METRICS: dict = {
        "model_name": "BiLSTM-BiGRU-VAE",
        "accuracy": 0.987,
        "precision": 0.980,
        "recall": 1.000,
        "f1_score": 0.990,
        "roc_auc": 0.999,
        "specificity": 0.984
    }

settings = Settings()

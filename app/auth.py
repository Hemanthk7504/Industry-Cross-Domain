from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from fastapi import Request, HTTPException, status
from app.config import settings
from app.database import SessionLocal
import hashlib

# Demo Users Store (In-memory fallback & cache)
USERS_DB: Dict[str, dict] = {
    "engineer@plant.com": {
        "email": "engineer@plant.com",
        "name": "Marcus Vance",
        "role": "engineer",
        "role_title": "Senior Maintenance Engineer",
        "department": "Mechanical Reliability Unit",
        "avatar": "MV",
        "password_hash": hashlib.sha256("plant123".encode()).hexdigest()
    },
    "lead@plant.com": {
        "email": "lead@plant.com",
        "name": "Dr. Elena Rostova",
        "role": "lead",
        "role_title": "Lead Reliability Specialist",
        "department": "Predictive Diagnostics & AI",
        "avatar": "ER",
        "password_hash": hashlib.sha256("plant123".encode()).hexdigest()
    },
    "manager@plant.com": {
        "email": "manager@plant.com",
        "name": "Arthur Pendelton",
        "role": "manager",
        "role_title": "Plant Operations Director",
        "department": "Smart Industrial Operations",
        "avatar": "AP",
        "password_hash": hashlib.sha256("plant123".encode()).hexdigest()
    },
    "auditor@plant.com": {
        "email": "auditor@plant.com",
        "name": "Clara Lindqvist",
        "role": "auditor",
        "role_title": "ISO 55001 Compliance Auditor",
        "department": "Quality Assurance & Standards",
        "avatar": "CL",
        "password_hash": hashlib.sha256("plant123".encode()).hexdigest()
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_user_by_email(email: str) -> Optional[dict]:
    """Retrieves a user by email from SQLite/Postgres database with in-memory caching."""
    email = email.lower().strip()
    if email in USERS_DB:
        return USERS_DB[email]
    
    db = SessionLocal()
    try:
        from app.db_models import UserDB
        db_user = db.query(UserDB).filter(UserDB.email == email).first()
        if db_user:
            u_dict = db_user.to_dict()
            USERS_DB[email] = u_dict
            return u_dict
    except Exception as e:
        print(f"[Auth DB Warning] Error fetching user {email}: {e}")
    finally:
        db.close()
        
    return None

def register_user(email: str, password: str, name: str, role: str = "engineer", department: str = "Industrial Operations") -> dict:
    """Registers a new user and persists directly to the database."""
    email = email.lower().strip()
    
    # Check if already exists in memory or DB
    existing = get_user_by_email(email)
    if existing:
        raise ValueError("An account with this email address already exists. Please sign in.")
    
    parts = [p for p in name.strip().split() if p]
    initials = ("".join([p[0].upper() for p in parts[:2]])) if parts else "US"
    
    role_titles = {
        "engineer": "Senior Maintenance Engineer",
        "lead": "Lead Reliability Specialist",
        "manager": "Plant Operations Director",
        "auditor": "ISO 55001 Compliance Auditor"
    }
    
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    title = role_titles.get(role, "Operational Engineer")
    dept = department.strip() if department else "Industrial Operations"

    new_user = {
        "email": email,
        "name": name.strip(),
        "role": role,
        "role_title": title,
        "department": dept,
        "avatar": initials,
        "password_hash": pw_hash
    }

    # Persist to SQLAlchemy database
    db = SessionLocal()
    try:
        from app.db_models import UserDB
        db_user = UserDB(
            email=email,
            name=name.strip(),
            role=role,
            role_title=title,
            department=dept,
            avatar=initials,
            password_hash=pw_hash
        )
        db.add(db_user)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[Auth DB Warning] Failed to persist user to database: {e}")
    finally:
        db.close()

    USERS_DB[email] = new_user
    return new_user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_current_user_from_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return get_user_by_email(email)
    except JWTError:
        return None

async def get_current_user_optional(request: Request) -> Optional[dict]:
    """
    Returns authenticated user if access_token cookie or Bearer header is present and valid.
    Returns None if unauthenticated (public guest mode - no sidebar).
    """
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if token:
        user = get_current_user_from_token(token)
        if user:
            return user
    return None

async def require_auth(request: Request) -> dict:
    """
    Enforces authentication for protected operational routes.
    Raises 401 if not logged in.
    """
    user = await get_current_user_optional(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access operational dashboard."
        )
    return user

async def get_current_user(request: Request) -> dict:
    return await require_auth(request)

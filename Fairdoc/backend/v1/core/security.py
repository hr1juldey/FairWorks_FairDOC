"""
V1 Enterprise Security Module
Medical-grade security with RBAC, master password, and JWT management
ALL SECURITY OPERATIONS GO THROUGH THIS MODULE
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
import string
import hashlib
import hmac
import os
import logging
from enum import Enum

from ..datamodels.auth_models import UserDB, UserSessionDB, UserRole, DoctorVerificationStatus
from ..data.database import get_db

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY CONFIGURATION - MEDICAL GRADE
# =============================================================================

class SecurityLevel(str, Enum):
    """Security levels for medical data protection"""
    PUBLIC = "public"           # No authentication required
    BASIC = "basic"            # Basic authentication
    MEDICAL = "medical"        # Medical data access
    EMERGENCY = "emergency"    # Emergency services access
    ADMIN = "admin"           # Administrative access
    DEVELOPER = "developer"   # Development access (god mode)


# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-medical-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Master Password for V1 Access (Additional Security Layer)
MASTER_PASSWORD_HASH = os.getenv("V1_MASTER_PASSWORD_HASH", "")  # Set in production
MASTER_PASSWORD_SALT = os.getenv("V1_MASTER_PASSWORD_SALT", "fairdoc_v1_medical_salt")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# =============================================================================
# MASTER PASSWORD PROTECTION (Medical Security Requirement)
# =============================================================================

def verify_master_password(password: str) -> bool:
    """Verify master password for V1 access"""
    if not MASTER_PASSWORD_HASH:
        # In development, allow bypass if no master password set
        return True
    
    # Create hash with salt
    salted_password = f"{password}{MASTER_PASSWORD_SALT}"
    password_hash = hashlib.pbkdf2_hmac('sha256',
                                       salted_password.encode('utf-8'),
                                       MASTER_PASSWORD_SALT.encode('utf-8'),
                                       100000)
    
    return hmac.compare_digest(password_hash.hex(), MASTER_PASSWORD_HASH)

def set_master_password(password: str) -> str:
    """Set master password and return hash (for production setup)"""
    salted_password = f"{password}{MASTER_PASSWORD_SALT}"
    password_hash = hashlib.pbkdf2_hmac('sha256',
                                       salted_password.encode('utf-8'),
                                       MASTER_PASSWORD_SALT.encode('utf-8'),
                                       100000)
    return password_hash.hex()

# =============================================================================
# PASSWORD AND TOKEN UTILITIES
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password with bcrypt"""
    return pwd_context.hash(password)

def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure token"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_api_key() -> str:
    """Generate secure API key"""
    return f"fai_v1_{generate_secure_token(40)}"

# =============================================================================
# JWT TOKEN MANAGEMENT
# =============================================================================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token with medical security claims"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Add security claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
        "api_version": "v1",
        "security_level": determine_security_level(data.get("role", "patient")),
        "medical_access": data.get("role") in ["doctor", "admin", "developer"]
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "api_version": "v1"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token with security validation"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Validate token type
        if payload.get("type") != token_type:
            logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
            return None
        
        # Validate API version
        if payload.get("api_version") != "v1":
            logger.warning(f"Invalid API version: {payload.get('api_version')}")
            return None
        
        # Check expiration
        if datetime.utcnow() > datetime.fromtimestamp(payload.get("exp", 0)):
            logger.warning("Token expired")
            return None
            
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

def determine_security_level(role: str) -> str:
    """Determine security level based on user role"""
    security_mapping = {
        "developer": SecurityLevel.DEVELOPER,
        "admin": SecurityLevel.ADMIN,
        "doctor": SecurityLevel.MEDICAL,
        "data_scientist": SecurityLevel.MEDICAL,
        "patient": SecurityLevel.BASIC
    }
    return security_mapping.get(role, SecurityLevel.BASIC).value

# =============================================================================
# USER AUTHENTICATION AND AUTHORIZATION
# =============================================================================

def authenticate_user(
    db: Session,
    username: str,
    password: str,
    master_password: Optional[str] = None
) -> Optional[UserDB]:
    """Authenticate user with master password check"""
    
    # Check master password first (medical security requirement)
    if master_password and not verify_master_password(master_password):
        logger.warning(f"Master password verification failed for user: {username}")
        return None
    
    # Find user by username or email
    user = (
        db.query(UserDB)
        .filter(
            (UserDB.username == username) | (UserDB.email == username)
        )
        .first()
    )
    
    if not user:
        logger.warning(f"User not found: {username}")
        return None
    
    # Verify password
    if not verify_password(password, user.hashed_password):
        # Increment failed attempts
        user.failed_login_attempts += 1
        db.commit()
        logger.warning(f"Failed login attempt for user: {username}")
        return None
    
    # Check if account is locked
    if user.failed_login_attempts >= 5:
        logger.warning(f"Account locked due to failed attempts: {username}")
        return None
    
    # Check user status
    if user.status not in ["active"]:
        logger.warning(f"Inactive user attempted login: {username}")
        return None
    
    # Reset failed attempts on successful login
    user.failed_login_attempts = 0
    user.last_login = datetime.utcnow()
    db.commit()
    
    logger.info(f"Successful authentication: {username} (role: {user.role.value})")
    return user

def get_user_permissions(user_role: UserRole) -> List[str]:
    """Get comprehensive permissions for user role"""
    
    # Medical-grade RBAC permissions
    role_permissions = {
        UserRole.DEVELOPER: [
            "*"  # God mode - all permissions
        ],
        UserRole.ADMIN: [
            "user:read", "user:write", "user:delete",
            "doctor:verify", "doctor:read", "doctor:write",
            "case:read", "case:write", "case:delete",
            "protocol:read", "protocol:write",
            "service:manage", "billing:manage",
            "frontend:marry", "backend:marry",
            "reports:read", "analytics:read",
            "emergency:view", "system:monitor"
        ],
        UserRole.DOCTOR: [
            "patient:read", "patient:write",
            "case:read", "case:write", "case:emergency",
            "protocol:read", "reports:receive",
            "emergency:call", "profile:read", "profile:write",
            "verification:request", "medical:diagnose",
            "prescription:write", "referral:create"
        ],
        UserRole.DATA_SCIENTIST: [
            "model:read", "model:write", "model:deploy",
            "data:read", "data:export", "analytics:read",
            "analytics:write", "service:read", "service:write",
            "ml:train", "ml:evaluate", "bias:monitor"
        ],
        UserRole.PATIENT: [
            "chat:ai", "chat:doctor", "chat:admin",
            "case:create", "case:read_own",
            "file:upload", "report:download_own",
            "complaint:create", "profile:read", "profile:write"
        ]
    }
    
    return role_permissions.get(user_role, [])

# =============================================================================
# DEPENDENCY FUNCTIONS FOR FASTAPI
# =============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[UserDB]:
    """Get current authenticated user from JWT token"""
    
    if not credentials:
        return None
    
    try:
        payload = verify_token(credentials.credentials, "access")
        if payload is None:
            return None
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        user = db.query(UserDB).filter(UserDB.user_id == user_id).first()
        if user is None:
            return None
        
        # Additional security checks for medical data
        if payload.get("medical_access") and user.role not in [
            UserRole.DOCTOR, UserRole.ADMIN, UserRole.DEVELOPER
        ]:
            logger.warning(f"Medical access attempted by non-medical user: {user.username}")
            return None
        
        return user
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return None

async def get_current_active_user(
    current_user: Optional[UserDB] = Depends(get_current_user)
) -> UserDB:
    """Get current active user with status validation"""
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if current_user.status not in ["active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account inactive or suspended"
        )
    
    return current_user

# =============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# =============================================================================

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: UserDB = Depends(get_current_active_user)):
        permissions = get_user_permissions(current_user.role)
        
        # Developer has all permissions
        if "*" in permissions:
            return current_user
        
        # Check specific permission
        if permission not in permissions:
            logger.warning(f"Permission denied: {current_user.username} lacks {permission}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission}"
            )
        
        return current_user
    return permission_checker

def require_role(role: UserRole):
    """Decorator to require specific role"""
    def role_checker(current_user: UserDB = Depends(get_current_active_user)):
        if current_user.role != role:
            logger.warning(f"Role access denied: {current_user.username} is {current_user.role.value}, required {role.value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {role.value}"
            )
        return current_user
    return role_checker

def require_security_level(level: SecurityLevel):
    """Decorator to require minimum security level"""
    def security_checker(current_user: UserDB = Depends(get_current_active_user)):
        user_level = determine_security_level(current_user.role.value)
        
        security_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.BASIC: 1,
            SecurityLevel.MEDICAL: 2,
            SecurityLevel.EMERGENCY: 3,
            SecurityLevel.ADMIN: 4,
            SecurityLevel.DEVELOPER: 5
        }
        
        required_level = security_hierarchy.get(level, 0)
        user_security_level = security_hierarchy.get(SecurityLevel(user_level), 0)
        
        if user_security_level < required_level:
            logger.warning(f"Security level denied: {current_user.username} has {user_level}, required {level.value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient security level. Required: {level.value}"
            )
        
        return current_user
    return security_checker

# =============================================================================
# ROLE-SPECIFIC DEPENDENCIES
# =============================================================================

async def get_developer_user(
    current_user: UserDB = Depends(require_role(UserRole.DEVELOPER))
) -> UserDB:
    """Get developer user (god mode)"""
    return current_user

async def get_admin_user(
    current_user: UserDB = Depends(require_role(UserRole.ADMIN))
) -> UserDB:
    """Get admin user"""
    return current_user

async def get_doctor_user(
    current_user: UserDB = Depends(require_role(UserRole.DOCTOR))
) -> UserDB:
    """Get verified doctor user"""
    # Additional check for doctor verification
    if not current_user.doctor_profile:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor profile not found"
        )
    
    if current_user.doctor_profile.verification_status != DoctorVerificationStatus.VERIFIED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor verification required"
        )
    
    return current_user

async def get_data_scientist_user(
    current_user: UserDB = Depends(require_role(UserRole.DATA_SCIENTIST))
) -> UserDB:
    """Get data scientist user"""
    return current_user

async def get_patient_user(
    current_user: UserDB = Depends(require_role(UserRole.PATIENT))
) -> UserDB:
    """Get patient user"""
    return current_user

async def get_medical_user(
    current_user: UserDB = Depends(require_security_level(SecurityLevel.MEDICAL))
) -> UserDB:
    """Get user with medical data access (doctor, admin, developer)"""
    return current_user

async def get_chat_user(
    current_user: UserDB = Depends(get_current_active_user)
) -> UserDB:
    """Get user who can participate in chat"""
    allowed_roles = [UserRole.PATIENT, UserRole.DOCTOR, UserRole.ADMIN, UserRole.DEVELOPER]
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chat access denied"
        )
    return current_user

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def create_user_session(
    db: Session,
    user: UserDB,
    request: Request,
    master_password_verified: bool = False
) -> UserSessionDB:
    """Create secure user session with device tracking"""
    
    # Create token data
    token_data = {
        "sub": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "permissions": get_user_permissions(user.role),
        "master_verified": master_password_verified
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": user.user_id})
    
    # Create session with device info
    session = UserSessionDB(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        device_info={
            "user_agent": request.headers.get("user-agent", ""),
            "accept_language": request.headers.get("accept-language", ""),
            "master_verified": master_password_verified
        },
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", "")
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    logger.info(f"Session created for user: {user.username} (session: {session.session_id})")
    return session

def invalidate_user_session(db: Session, session_id: str) -> bool:
    """Invalidate user session"""
    session = db.query(UserSessionDB).filter(UserSessionDB.session_id == session_id).first()
    if session:
        session.is_active = False
        db.commit()
        logger.info(f"Session invalidated: {session_id}")
        return True
    return False

def invalidate_all_user_sessions(db: Session, user_id: int) -> int:
    """Invalidate all sessions for a user (for security incidents)"""
    updated = db.query(UserSessionDB).filter(
        UserSessionDB.user_id == user_id,
        UserSessionDB.is_active
    ).update({"is_active": False})
    
    db.commit()
    logger.info(f"Invalidated {updated} sessions for user ID: {user_id}")
    return updated

# =============================================================================
# SECURITY MONITORING AND LOGGING
# =============================================================================

def log_security_event(
    event_type: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Log security events for monitoring"""
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "ip_address": ip_address,
        "details": details or {},
        "api_version": "v1"
    }
    
    # In production, send to security monitoring system
    logger.warning(f"SECURITY EVENT: {log_entry}")

def check_rate_limit(
    request: Request,
    user_id: Optional[str] = None,
    limit: int = 100,
    window: int = 60
) -> bool:
    """Check rate limit for security"""
    # Simplified rate limiting - in production use Redis
    # For now, just log the request
    
    client_ip = request.client.host if request.client else "unknown"
    log_security_event(
        event_type="api_request",
        user_id=user_id,
        ip_address=client_ip,
        details={"endpoint": str(request.url), "method": request.method}
    )
    
    return True  # Allow all requests in MVP

# =============================================================================
# EMERGENCY ACCESS FUNCTIONS
# =============================================================================

def emergency_access_override(
    emergency_code: str,
    user_id: str,
    reason: str
) -> bool:
    """Emergency access override for critical medical situations"""
    
    # Verify emergency code (in production, use secure emergency codes)
    emergency_codes = os.getenv("EMERGENCY_CODES", "").split(",")
    
    if emergency_code not in emergency_codes:
        log_security_event(
            event_type="emergency_access_denied",
            user_id=user_id,
            details={"reason": reason, "invalid_code": True}
        )
        return False
    
    log_security_event(
        event_type="emergency_access_granted",
        user_id=user_id,
        details={"reason": reason, "emergency_code_used": True}
    )
    
    return True

# =============================================================================
# EXPORT
# =============================================================================


__all__ = [
    # Security levels
    "SecurityLevel",
    
    # Master password
    "verify_master_password", "set_master_password",
    
    # Password utilities
    "verify_password", "get_password_hash", "generate_secure_token", "generate_api_key",
    
    # JWT utilities
    "create_access_token", "create_refresh_token", "verify_token",
    
    # Authentication
    "authenticate_user", "get_user_permissions",
    
    # User dependencies
    "get_current_user", "get_current_active_user",
    
    # RBAC dependencies
    "require_permission", "require_role", "require_security_level",
    "get_developer_user", "get_admin_user", "get_doctor_user",
    "get_data_scientist_user", "get_patient_user", "get_medical_user", "get_chat_user",
    
    # Session management
    "create_user_session", "invalidate_user_session", "invalidate_all_user_sessions",
    
    # Security monitoring
    "log_security_event", "check_rate_limit", "emergency_access_override"
]

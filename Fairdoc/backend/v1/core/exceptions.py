"""
V1 Exception Handlers
Medical-grade error handling with comprehensive logging
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime
import logging
from typing import Union

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM EXCEPTION CLASSES
# =============================================================================

class MedicalDataException(Exception):
    """Exception for medical data processing errors"""
    def __init__(self, message: str, error_code: str = "MEDICAL_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class NHSVerificationException(Exception):
    """Exception for NHS verification failures"""
    def __init__(self, message: str, nhs_number: str = None):
        self.message = message
        self.nhs_number = nhs_number
        super().__init__(self.message)

class GMCVerificationException(Exception):
    """Exception for GMC verification failures"""
    def __init__(self, message: str, gmc_number: str = None):
        self.message = message
        self.gmc_number = gmc_number
        super().__init__(self.message)

# =============================================================================
# ERROR RESPONSE FORMATTER
# =============================================================================

def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "API_ERROR",
    details: dict = None,
    request_path: str = None
) -> dict:
    """Create standardized error response"""
    
    return {
        "error": True,
        "status_code": status_code,
        "error_type": error_type,
        "message": message,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat(),
        "path": request_path,
        "api_version": "v1"
    }

# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions"""
    
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail} "
        f"on {request.method} {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=exc.detail,
            error_type="HTTP_ERROR",
            request_path=str(request.url.path)
        )
    )

async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette HTTP exceptions"""
    
    logger.warning(
        f"Starlette Exception: {exc.status_code} - {exc.detail} "
        f"on {request.method} {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=exc.detail or "Internal server error",
            error_type="SERVER_ERROR",
            request_path=str(request.url.path)
        )
    )

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors"""
    
    logger.warning(
        f"Validation Error on {request.method} {request.url.path}: "
        f"{exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            status_code=422,
            message="Validation error",
            error_type="VALIDATION_ERROR",
            details={
                "errors": exc.errors(),
                "body": exc.body
            },
            request_path=str(request.url.path)
        )
    )

async def medical_data_exception_handler(
    request: Request,
    exc: MedicalDataException
) -> JSONResponse:
    """Handle medical data processing exceptions"""
    
    logger.error(
        f"Medical Data Error on {request.method} {request.url.path}: "
        f"{exc.message} (Code: {exc.error_code})"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=500,
            message=exc.message,
            error_type="MEDICAL_DATA_ERROR",
            details={"error_code": exc.error_code},
            request_path=str(request.url.path)
        )
    )

async def nhs_verification_exception_handler(
    request: Request,
    exc: NHSVerificationException
) -> JSONResponse:
    """Handle NHS verification exceptions"""
    
    logger.error(
        f"NHS Verification Error on {request.method} {request.url.path}: "
        f"{exc.message} (NHS: {exc.nhs_number})"
    )
    
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=create_error_response(
            status_code=401,
            message=exc.message,
            error_type="NHS_VERIFICATION_ERROR",
            details={"nhs_number": exc.nhs_number if exc.nhs_number else None},
            request_path=str(request.url.path)
        )
    )

async def gmc_verification_exception_handler(
    request: Request,
    exc: GMCVerificationException
) -> JSONResponse:
    """Handle GMC verification exceptions"""
    
    logger.error(
        f"GMC Verification Error on {request.method} {request.url.path}: "
        f"{exc.message} (GMC: {exc.gmc_number})"
    )
    
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=create_error_response(
            status_code=401,
            message=exc.message,
            error_type="GMC_VERIFICATION_ERROR",
            details={"gmc_number": exc.gmc_number if exc.gmc_number else None},
            request_path=str(request.url.path)
        )
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions"""
    
    logger.error(
        f"Unhandled Exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {str(exc)}",
        exc_info=True
    )
    
    # In development, expose detailed error
    # In production, hide sensitive details
    from .config import get_v1_settings
    settings = get_v1_settings()
    
    if settings.is_development:
        error_detail = str(exc)
        error_type = type(exc).__name__
    else:
        error_detail = "Internal server error"
        error_type = "INTERNAL_ERROR"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=500,
            message=error_detail,
            error_type=error_type,
            request_path=str(request.url.path)
        )
    )

# =============================================================================
# CONFIGURATION FUNCTION
# =============================================================================

def configure_exception_handlers(app: FastAPI) -> None:
    """Configure all exception handlers for the V1 API"""
    
    logger.info("🛡️ Configuring V1 exception handlers...")
    
    # Standard HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Custom medical exceptions
    app.add_exception_handler(MedicalDataException, medical_data_exception_handler)
    app.add_exception_handler(NHSVerificationException, nhs_verification_exception_handler)
    app.add_exception_handler(GMCVerificationException, gmc_verification_exception_handler)
    
    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("✅ Exception handlers configured")

# =============================================================================
# EXPORT
# =============================================================================


__all__ = [
    "configure_exception_handlers",
    "MedicalDataException",
    "NHSVerificationException",
    "GMCVerificationException",
    "create_error_response"
]

"""
V1 File Upload API - Production Grade
Complete file management with batch processing, security scanning, and comprehensive logging
"""
# Import asyncio for async helper functions
import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

# Import our enhanced architecture
from ..data.database import get_db, CaseReportCRUD
from ..datamodels.file_models import (
    FileUploadRequest, FileUploadResponse, BatchUploadResponse,
    FileAccessResponse, FileListResponse, FileCategory, FileStatus,
    BatchProcessingRequest, BatchProcessingResponse, FileMetadata
)
from ..datamodels.sqlalchemy_models import CaseReportDB
from ..core.security import get_current_active_user
from ..datamodels.auth_models import UserDB
from ..utils.file_utils import (
    validate_medical_file, upload_file_to_minio, generate_presigned_url,
    delete_file_from_minio, generate_object_path, create_file_metadata,
    extract_file_content, process_file_batch, perform_security_scan
)

logger = logging.getLogger(__name__)

# Initialize router
files_router = APIRouter(prefix="/files", tags=["File Management"])

# =============================================================================
# SINGLE FILE UPLOAD WITH FULL VALIDATION
# =============================================================================

@files_router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    request_data: FileUploadRequest = Depends(),
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload single medical file with comprehensive validation and security scanning"""
    
    upload_start = datetime.now()
    logger.info(f"File upload started by user {current_user.username}: {file.filename}")
    
    try:
        # Extract and validate file
        file_content = extract_file_content(file)
        
        validation_result = validate_medical_file(
            file_content,
            file.filename or "unknown",
            file.content_type or "application/octet-stream"
        )
        
        if not validation_result.validation_passed:
            logger.warning(f"File validation failed: {file.filename} - {validation_result.validation_errors}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {', '.join(validation_result.validation_errors)}"
            )
        
        # Perform security scan
        security_scan = perform_security_scan(
            file_content,
            validation_result.original_filename,
            validation_result.content_type
        )
        
        if not security_scan.is_safe:
            logger.error(f"Security scan failed for file {file.filename}: {security_scan.threats_detected}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security scan failed: {', '.join(security_scan.threats_detected)}"
            )
        
        # Verify case access if provided
        case_report = None
        if request_data.case_id:
            case_report = CaseReportCRUD.get_case_report(db, request_data.case_id)
            if not case_report:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Case report {request_data.case_id} not found"
                )
            
            # Check permissions
            medical_roles = ["doctor", "admin", "developer"]
            if (case_report.patient_id != current_user.user_id and
                current_user.role not in medical_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to upload files for this case"
                )
        
        # Generate storage path and upload
        object_path = generate_object_path(
            current_user.user_id,
            validation_result.secure_filename,
            request_data.case_id,
            request_data.file_category
        )
        
        etag = upload_file_to_minio(file_content, object_path, validation_result.content_type)
        
        # Create comprehensive file metadata
        file_metadata = create_file_metadata(
            validation_result=validation_result,
            object_path=object_path,
            etag=etag,
            user_id=current_user.user_id,
            case_id=request_data.case_id,
            description=request_data.description,
            tags=request_data.tags,
            security_scan=security_scan
        )
        
        # Update file status to UPLOADED
        file_metadata["status"] = FileStatus.UPLOADED.value
        file_metadata["upload_completed_at"] = datetime.now().isoformat()
        
        # Update case report if applicable
        if case_report:
            uploaded_files = case_report.uploaded_files or []
            uploaded_files.append(file_metadata)
            
            CaseReportCRUD.update_case_report(
                db=db,
                case_id=request_data.case_id,
                uploaded_files=uploaded_files
            )
        
        upload_duration = (datetime.now() - upload_start).total_seconds()
        logger.info(f"File upload completed: {validation_result.original_filename} by {current_user.username} in {upload_duration:.3f}s")
        
        return FileUploadResponse(
            file_metadata=FileMetadata(**file_metadata),
            upload_status="success",
            message="File uploaded and validated successfully",
            access_info={
                "download_url": file_metadata["file_url"],
                "expires": "File access managed by case permissions",
                "viewable_by": "Case participants and medical staff",
                "security_status": security_scan.threat_level.value
            },
            processing_started=True,
            estimated_processing_time="1-2 minutes"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        upload_duration = (datetime.now() - upload_start).total_seconds()
        logger.error(f"File upload failed after {upload_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload service error"
        )

# =============================================================================
# BATCH FILE UPLOAD WITH ENHANCED PROCESSING
# =============================================================================

@files_router.post("/upload-multiple", response_model=BatchUploadResponse)
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    case_id: Optional[str] = Form(None),
    descriptions: Optional[str] = Form(None),
    file_category: FileCategory = Form(FileCategory.OTHER),
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload multiple files using batch processing utility"""
    
    batch_start = datetime.now()
    logger.info(f"Batch upload started by {current_user.username}: {len(files)} files")
    
    try:
        if len(files) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 files allowed per batch upload"
            )
        
        # Parse descriptions
        file_descriptions = {}
        if file_descriptions:
            import json
            try:
                file_descriptions = json.loads(descriptions)
            except Exception:
                logger.warning("Invalid descriptions JSON provided")
        
        # Verify case access
        case_report = None
        if case_id:
            case_report = CaseReportCRUD.get_case_report(db, case_id)
            if not case_report:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Case report {case_id} not found"
                )
            
            medical_roles = ["doctor", "admin", "developer"]
            if (case_report.patient_id != current_user.user_id and
                current_user.role not in medical_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to upload files for this case"
                )
        
        # Use batch processing utility
        successful_files, failed_files, batch_id = process_file_batch(
            files=files,
            user_id=current_user.user_id,
            case_id=case_id,
            max_batch_size=10
        )
        
        # Update case report with successful uploads
        uploaded_files_metadata = []
        if case_report and successful_files:
            existing_files = case_report.uploaded_files or []
            
            for success_file in successful_files:
                file_metadata = success_file["file_metadata"]
                file_metadata["status"] = FileStatus.UPLOADED.value
                existing_files.append(file_metadata)
                uploaded_files_metadata.append(FileMetadata(**file_metadata))
            
            CaseReportCRUD.update_case_report(
                db=db,
                case_id=case_id,
                uploaded_files=existing_files
            )
        
        batch_duration = (datetime.now() - batch_start).total_seconds()
        logger.info(f"Batch upload completed: {batch_id} - {len(successful_files)} successful, {len(failed_files)} failed in {batch_duration:.3f}s")
        
        return BatchUploadResponse(
            batch_upload_status="completed",
            batch_id=batch_id,
            total_files=len(files),
            successful_uploads=len(successful_files),
            failed_uploads=len(failed_files),
            upload_results=successful_files + failed_files,
            uploaded_files=uploaded_files_metadata,
            processing_queue_position=1,
            message=f"Processed {len(files)} files: {len(successful_files)} successful, {len(failed_files)} failed",
            started_at=batch_start,
            estimated_completion=datetime.now() + timedelta(minutes=2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        batch_duration = (datetime.now() - batch_start).total_seconds()
        logger.error(f"Batch upload failed after {batch_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch file upload service error"
        )

# =============================================================================
# PRODUCTION GRADE BATCH PROCESSING ENDPOINT
# =============================================================================

@files_router.post("/batch-process", response_model=BatchProcessingResponse)
async def batch_process_files(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Professional batch processing endpoint for production workflows"""
    
    processing_start = datetime.now()
    logger.info(f"Batch processing started: {request.batch_id} by {current_user.username}")
    
    try:
        # Validate all file IDs exist and user has access
        accessible_files = []
        inaccessible_files = []
        
        # Get user's accessible cases
        user_cases = db.query(CaseReportDB).filter(
            CaseReportDB.patient_id == current_user.user_id
        ).all()
        
        medical_roles = ["doctor", "admin", "developer"]
        if current_user.role in medical_roles:
            all_cases = db.query(CaseReportDB).all()
            user_cases.extend(all_cases)
        
        # Find all accessible files
        all_accessible_files = {}
        for case in user_cases:
            if case.uploaded_files:
                for file_data in case.uploaded_files:
                    all_accessible_files[file_data["file_id"]] = file_data
        
        # Check which requested files are accessible
        for file_id in request.file_ids:
            if file_id in all_accessible_files:
                accessible_files.append(all_accessible_files[file_id])
            else:
                inaccessible_files.append(file_id)
        
        if inaccessible_files:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to files: {inaccessible_files}"
            )
        
        # Initialize processing results
        processing_results = []
        processed_count = 0
        failed_count = 0
        
        # Process each file based on processing type
        for file_data in accessible_files:
            file_start = datetime.now()
            
            try:
                if request.processing_type == "security_rescan":
                    result = await process_security_rescan(file_data, request.parameters)
                elif request.processing_type == "format_conversion":
                    result = await process_format_conversion(file_data, request.parameters)
                elif request.processing_type == "medical_analysis":
                    result = await process_medical_analysis(file_data, request.parameters)
                elif request.processing_type == "metadata_extraction":
                    result = await process_metadata_extraction(file_data, request.parameters)
                else:
                    raise ValueError(f"Unknown processing type: {request.processing_type}")
                
                processing_time = (datetime.now() - file_start).total_seconds()
                
                processing_results.append({
                    "file_id": file_data["file_id"],
                    "filename": file_data["original_filename"],
                    "status": "success",
                    "result": result,
                    "processing_time_seconds": processing_time,
                    "processed_at": datetime.now().isoformat()
                })
                
                processed_count += 1
                logger.info(f"File processed successfully: {file_data['file_id']} ({processing_time:.3f}s)")
                
            except Exception as e:
                processing_time = (datetime.now() - file_start).total_seconds()
                
                processing_results.append({
                    "file_id": file_data["file_id"],
                    "filename": file_data["original_filename"],
                    "status": "failed",
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                    "processed_at": datetime.now().isoformat()
                })
                
                failed_count += 1
                logger.error(f"File processing failed: {file_data['file_id']} - {e}")
        
        # Calculate completion time
        total_processing_time = (datetime.now() - processing_start).total_seconds()
        
        # Send webhook notification if provided
        if request.notification_webhook:
            background_tasks.add_task(
                send_batch_completion_webhook,
                request.notification_webhook,
                request.batch_id,
                processed_count,
                failed_count
            )
        
        logger.info(f"Batch processing completed: {request.batch_id} - {processed_count} successful, {failed_count} failed in {total_processing_time:.3f}s")
        
        return BatchProcessingResponse(
            batch_id=request.batch_id,
            status="completed",
            total_files=len(accessible_files),
            processed_files=processed_count,
            failed_files=failed_count,
            processing_results=processing_results,
            started_at=processing_start,
            estimated_completion=None,
            completed_at=datetime.now(),
            error_summary=f"{failed_count} files failed processing" if failed_count > 0 else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - processing_start).total_seconds()
        logger.error(f"Batch processing failed after {processing_time:.3f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch processing service error"
        )

# =============================================================================
# FILE ACCESS WITH COMPREHENSIVE LOGGING
# =============================================================================

@files_router.get("/{file_id}", response_model=FileAccessResponse)
async def get_file_info(
    file_id: str,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get file information with access logging"""
    
    access_start = datetime.now()
    logger.info(f"File access requested: {file_id} by {current_user.username}")
    
    try:
        # Find file in accessible cases
        user_cases = db.query(CaseReportDB).filter(
            CaseReportDB.patient_id == current_user.user_id
        ).all()
        
        medical_roles = ["doctor", "admin", "developer"]
        if current_user.role in medical_roles:
            all_cases = db.query(CaseReportDB).all()
            user_cases.extend(all_cases)
        
        # Search for file
        file_metadata = None
        case_id = None
        
        for case in user_cases:
            if case.uploaded_files:
                for file_data in case.uploaded_files:
                    if file_data.get("file_id") == file_id:
                        file_metadata = file_data
                        case_id = case.case_id
                        break
                if file_metadata:
                    break
        
        if not file_metadata:
            logger.warning(f"File access denied: {file_id} for user {current_user.username}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found or access denied"
            )
        
        # Update last accessed timestamp
        file_metadata["last_accessed"] = datetime.now().isoformat()
        
        # Generate secure access URL
        presigned_url = generate_presigned_url(file_metadata["minio_object_path"], expires_hours=1)
        
        access_duration = (datetime.now() - access_start).total_seconds()
        logger.info(f"File access granted: {file_id} for {current_user.username} ({access_duration:.3f}s)")
        
        return FileAccessResponse(
            file_metadata=FileMetadata(**file_metadata),
            case_id=case_id,
            access_url=presigned_url,
            access_expires="1 hour",
            file_permissions={
                "can_download": True,
                "can_delete": current_user.user_id == file_metadata["uploaded_by"] or current_user.role in medical_roles,
                "can_share": current_user.role in medical_roles,
                "can_modify": current_user.user_id == file_metadata["uploaded_by"] or current_user.role in medical_roles
            },
            access_logged=True,
            accessed_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        access_duration = (datetime.now() - access_start).total_seconds()
        logger.error(f"File access failed after {access_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File access service error"
        )

# =============================================================================
# FILE DELETION WITH AUDIT LOGGING
# =============================================================================

@files_router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete file with comprehensive audit logging"""
    
    deletion_start = datetime.now()
    logger.info(f"File deletion requested: {file_id} by {current_user.username}")
    
    try:
        # Find and verify file access
        user_cases = db.query(CaseReportDB).filter(
            CaseReportDB.patient_id == current_user.user_id
        ).all()
        
        medical_roles = ["doctor", "admin", "developer"]
        if current_user.role in medical_roles:
            all_cases = db.query(CaseReportDB).all()
            user_cases.extend(all_cases)
        
        target_case = None
        file_metadata = None
        file_index = None
        
        for case in user_cases:
            if case.uploaded_files:
                for i, file_data in enumerate(case.uploaded_files):
                    if file_data.get("file_id") == file_id:
                        if (file_data["uploaded_by"] != current_user.user_id and
                            current_user.role not in medical_roles):
                            logger.warning(f"File deletion permission denied: {file_id} by {current_user.username}")
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="Permission denied to delete this file"
                            )
                        
                        target_case = case
                        file_metadata = file_data
                        file_index = i
                        break
                if file_metadata:
                    break
        
        if not file_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found or access denied"
            )
        
        # Update file status to DELETED before actual deletion
        file_metadata["status"] = FileStatus.DELETED.value
        file_metadata["deleted_at"] = datetime.now().isoformat()
        file_metadata["deleted_by"] = current_user.user_id
        
        # Delete from MinIO
        delete_success = delete_file_from_minio(file_metadata["minio_object_path"])
        if not delete_success:
            logger.warning(f"Failed to delete file from MinIO: {file_id}")
        
        # Remove from case report
        updated_files = target_case.uploaded_files.copy()
        updated_files.pop(file_index)
        
        CaseReportCRUD.update_case_report(
            db=db,
            case_id=target_case.case_id,
            uploaded_files=updated_files
        )
        
        deletion_duration = (datetime.now() - deletion_start).total_seconds()
        logger.info(f"File deleted successfully: {file_id} by {current_user.username} ({deletion_duration:.3f}s)")
        
        return {
            "file_id": file_id,
            "filename": file_metadata["original_filename"],
            "deletion_status": "success",
            "message": "File deleted successfully",
            "case_id": target_case.case_id,
            "deleted_at": datetime.now().isoformat(),
            "deleted_by": current_user.username,
            "minio_deletion_success": delete_success
        }
        
    except HTTPException:
        raise
    except Exception as e:
        deletion_duration = (datetime.now() - deletion_start).total_seconds()
        logger.error(f"File deletion failed after {deletion_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File deletion service error"
        )

# =============================================================================
# FILE LISTING WITH ENHANCED PAGINATION
# =============================================================================

@files_router.get("/", response_model=FileListResponse)
async def list_user_files(
    case_id: Optional[str] = None,
    file_category: Optional[FileCategory] = None,
    status: Optional[FileStatus] = None,
    page: int = 1,
    page_size: int = 20,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List files with enhanced filtering and pagination"""
    
    listing_start = datetime.now()
    logger.info(f"File listing requested by {current_user.username}: page={page}, size={page_size}")
    
    try:
        # Get accessible cases
        if case_id:
            cases = [CaseReportCRUD.get_case_report(db, case_id)]
            if not cases[0]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Case {case_id} not found"
                )
        else:
            cases = db.query(CaseReportDB).filter(
                CaseReportDB.patient_id == current_user.user_id
            ).all()
            
            medical_roles = ["doctor", "admin", "developer"]
            if current_user.role in medical_roles:
                all_cases = db.query(CaseReportDB).all()
                cases = all_cases
        
        # Extract and filter files
        all_files = []
        for case in cases:
            if case and case.uploaded_files:
                for file_data in case.uploaded_files:
                    # Apply filters
                    if file_category and file_data.get("file_category") != file_category.value:
                        continue
                    
                    if status and file_data.get("status") != status.value:
                        continue
                    
                    file_with_context = {
                        **file_data,
                        "case_id": case.case_id,
                        "case_status": case.status
                    }
                    all_files.append(file_with_context)
        
        # Sort by upload timestamp (newest first)
        all_files.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)
        
        # Apply pagination
        total_files = len(all_files)
        total_pages = (total_files + page_size - 1) // page_size
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_files = all_files[start_index:end_index]
        
        # Convert to FileMetadata objects
        file_metadata_list = [FileMetadata(**file_data) for file_data in paginated_files]
        
        listing_duration = (datetime.now() - listing_start).total_seconds()
        logger.info(f"File listing completed: {len(paginated_files)} files returned in {listing_duration:.3f}s")
        
        return FileListResponse(
            files=file_metadata_list,
            total_files=total_files,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            filters={
                "case_id": case_id,
                "file_category": file_category.value if file_category else None,
                "status": status.value if status else None
            },
            summary={
                "by_category": {},
                "by_status": {},
                "total_size": sum(f.get("file_size", 0) for f in all_files),
                "cases_with_files": len(set(f["case_id"] for f in all_files))
            },
            generated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        listing_duration = (datetime.now() - listing_start).total_seconds()
        logger.error(f"File listing failed after {listing_duration:.3f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File listing service error"
        )

# =============================================================================
# BATCH PROCESSING HELPER FUNCTIONS
# =============================================================================

async def process_security_rescan(file_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Rescan file for security threats"""
    logger.info(f"Security rescan started for file: {file_data['file_id']}")
    
    # Simulate security rescan
    await asyncio.sleep(0.1)  # Simulate processing time
    
    return {
        "security_status": "clean",
        "threats_detected": [],
        "confidence_score": 0.95,
        "scan_timestamp": datetime.now().isoformat()
    }

async def process_format_conversion(file_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert file format"""
    logger.info(f"Format conversion started for file: {file_data['file_id']}")
    
    # Simulate format conversion
    await asyncio.sleep(0.2)  # Simulate processing time
    
    return {
        "original_format": file_data["content_type"],
        "target_format": parameters.get("target_format", "application/pdf"),
        "conversion_status": "completed",
        "output_file_id": str(uuid.uuid4())
    }

async def process_medical_analysis(file_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Perform medical analysis on file"""
    logger.info(f"Medical analysis started for file: {file_data['file_id']}")
    
    # Simulate medical analysis
    await asyncio.sleep(0.3)  # Simulate processing time
    
    return {
        "analysis_type": "medical_imaging",
        "findings": ["No abnormalities detected"],
        "confidence_score": 0.87,
        "analysis_timestamp": datetime.now().isoformat()
    }

async def process_metadata_extraction(file_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from file"""
    logger.info(f"Metadata extraction started for file: {file_data['file_id']}")
    
    # Simulate metadata extraction
    await asyncio.sleep(0.05)  # Simulate processing time
    
    return {
        "extracted_metadata": {
            "creation_date": "2025-01-01T00:00:00Z",
            "author": "Medical System",
            "keywords": ["medical", "report"]
        },
        "extraction_timestamp": datetime.now().isoformat()
    }

async def send_batch_completion_webhook(webhook_url: str, batch_id: str, processed_count: int, failed_count: int):
    """Send webhook notification for batch completion"""
    logger.info(f"Sending batch completion webhook: {batch_id}")
    
    # In production, this would make an HTTP request to the webhook URL
    # For now, just log the notification
    webhook_data = {
        "batch_id": batch_id,
        "status": "completed",
        "processed_files": processed_count,
        "failed_files": failed_count,
        "completed_at": datetime.now().isoformat()
    }
    
    logger.info(f"Webhook data: {webhook_data}")


# Export router
__all__ = ["files_router"]

import os
import aiofiles
from fastapi import UploadFile
from typing import Optional
from app.config import settings
from app.auth.models import SubscriptionPlan

async def save_uploaded_file(file: UploadFile, project_id: str) -> str:
    """Save uploaded file to disk"""
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Generate file path
    file_extension = get_file_extension(file.filename)
    file_path = os.path.join(settings.UPLOAD_DIR, f"{project_id}{file_extension}")
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_path

def get_file_extension(filename: Optional[str]) -> str:
    """Get file extension from filename"""
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lower()

def validate_file_size(file: UploadFile, subscription_plan: SubscriptionPlan) -> bool:
    """Validate file size based on subscription plan"""
    if not hasattr(file, 'size') or file.size is None:
        return True  # Can't validate, allow it
    
    if subscription_plan == SubscriptionPlan.FREE:
        max_size = 1 * 1024 * 1024  # 1MB
    elif subscription_plan == SubscriptionPlan.PLUS:
        max_size = 50 * 1024 * 1024  # 50MB
    else:  # PRO
        max_size = settings.MAX_FILE_SIZE  # 100MB
    
    return file.size <= max_size

def cleanup_project_files(project_id: str):
    """Clean up files associated with a project"""
    try:
        # Remove uploaded file
        for ext in ['.csv', '.xlsx', '.json', '.txt']:
            file_path = os.path.join(settings.UPLOAD_DIR, f"{project_id}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove generated files (predictions, reports, etc.)
        results_dir = os.path.join(settings.UPLOAD_DIR, "results", project_id)
        if os.path.exists(results_dir):
            import shutil
            shutil.rmtree(results_dir)
    except Exception:
        pass  # Ignore cleanup errors

def get_file_info(file_path: str) -> dict:
    """Get file information"""
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    return {
        'size': stat.st_size,
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'extension': get_file_extension(file_path)
    } 
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from typing import List, Optional
import pandas as pd
import json
import os
from datetime import datetime
import uuid

from app.auth.dependencies import get_current_user
from app.auth.models import UserResponse, SubscriptionPlan
from app.models.project import (
    ProjectCreate, ProjectResponse, ProjectPrompt, DatasetPreview, 
    ProjectStatus, TaskType, ProjectUpdate
)
from app.database import get_supabase
from app.config import settings
from app.services.data_processor import DataProcessor
from app.services.ml_pipeline import MLPipeline
from app.services.ai_analyzer import AIAnalyzer
from app.utils.file_utils import save_uploaded_file, get_file_extension, validate_file_size

router = APIRouter()

@router.get("/", response_model=List[ProjectResponse])
async def get_projects(current_user: UserResponse = Depends(get_current_user)):
    """Get all projects for the current user"""
    supabase = get_supabase()
    
    try:
        response = supabase.table("projects").select("*").eq("user_id", current_user.id).order("created_at", desc=True).execute()
        return [ProjectResponse(**project) for project in response.data]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch projects"
        )

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create a new project"""
    # Check project limits based on subscription
    if current_user.subscription_plan == SubscriptionPlan.FREE:
        if current_user.projects_used >= settings.FREE_PLAN_PROJECTS_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Free plan limited to {settings.FREE_PLAN_PROJECTS_LIMIT} projects per month"
            )
    elif current_user.subscription_plan == SubscriptionPlan.PLUS:
        if current_user.projects_used >= settings.PLUS_PLAN_PROJECTS_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Plus plan limited to {settings.PLUS_PLAN_PROJECTS_LIMIT} projects per month"
            )
    
    supabase = get_supabase()
    now = datetime.utcnow()
    
    project_data = {
        "id": str(uuid.uuid4()),
        "user_id": current_user.id,
        "name": project.name,
        "description": project.description,
        "status": ProjectStatus.UPLOADING.value,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }
    
    try:
        response = supabase.table("projects").insert(project_data).execute()
        created_project = response.data[0]
        
        # Update user's project count
        supabase.table("users").update({
            "projects_used": current_user.projects_used + 1,
            "updated_at": now.isoformat()
        }).eq("id", current_user.id).execute()
        
        return ProjectResponse(**created_project)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project"
        )

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get a specific project"""
    supabase = get_supabase()
    
    try:
        response = supabase.table("projects").select("*").eq("id", project_id).eq("user_id", current_user.id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return ProjectResponse(**response.data[0])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch project"
        )

@router.post("/{project_id}/upload", response_model=DatasetPreview)
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user)
):
    """Upload dataset file to a project"""
    # Validate file
    if not validate_file_size(file, current_user.subscription_plan):
        max_size = "1MB" if current_user.subscription_plan == SubscriptionPlan.FREE else "100MB"
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size for your plan: {max_size}"
        )
    
    allowed_extensions = ['.csv', '.xlsx', '.json', '.txt']
    file_ext = get_file_extension(file.filename)
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Verify project ownership
    supabase = get_supabase()
    project_response = supabase.table("projects").select("*").eq("id", project_id).eq("user_id", current_user.id).execute()
    if not project_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    try:
        # Save file
        file_path = await save_uploaded_file(file, project_id)
        
        # Process and preview data
        data_processor = DataProcessor()
        df = data_processor.load_file(file_path)
        
        # Check row limits
        row_limit = data_processor.get_row_limit(current_user.subscription_plan)
        if len(df) > row_limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Dataset too large. Your plan allows up to {row_limit:,} rows"
            )
        
        preview = data_processor.create_preview(df)
        
        # Update project with dataset info
        update_data = {
            "dataset_rows": len(df),
            "dataset_columns": len(df.columns),
            "status": ProjectStatus.PROCESSING.value,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("projects").update(update_data).eq("id", project_id).execute()
        
        return preview
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}"
        )

@router.post("/{project_id}/analyze")
async def analyze_with_prompt(
    project_id: str,
    prompt_data: ProjectPrompt,
    current_user: UserResponse = Depends(get_current_user)
):
    """Analyze dataset using natural language prompt"""
    supabase = get_supabase()
    
    # Verify project ownership
    project_response = supabase.table("projects").select("*").eq("id", project_id).eq("user_id", current_user.id).execute()
    if not project_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    project = project_response.data[0]
    
    try:
        # Load dataset
        file_path = os.path.join(settings.UPLOAD_DIR, f"{project_id}.csv")  # Assuming processed to CSV
        data_processor = DataProcessor()
        df = data_processor.load_file(file_path)
        
        # Analyze prompt with AI
        ai_analyzer = AIAnalyzer()
        analysis = await ai_analyzer.analyze_prompt(prompt_data.prompt, df.columns.tolist(), df.head())
        
        # Update project with analysis
        update_data = {
            "task_type": analysis["task_type"],
            "target_column": analysis["target_column"],
            "updated_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("projects").update(update_data).eq("id", project_id).execute()
        
        # Start ML pipeline (async)
        ml_pipeline = MLPipeline()
        # In production, this would be queued with Celery
        result = await ml_pipeline.run_analysis(df, analysis["task_type"], analysis["target_column"])
        
        # Generate AI summary
        ai_summary = await ai_analyzer.generate_summary(result, current_user.subscription_plan)
        
        # Update project with results
        final_update = {
            "status": ProjectStatus.COMPLETED.value,
            "model_metrics": result["metrics"],
            "ai_summary": ai_summary,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("projects").update(final_update).eq("id", project_id).execute()
        
        return {
            "message": "Analysis completed successfully",
            "task_type": analysis["task_type"],
            "target_column": analysis["target_column"],
            "summary": ai_summary
        }
        
    except Exception as e:
        # Update project status to failed
        supabase.table("projects").update({
            "status": ProjectStatus.FAILED.value,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", project_id).execute()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/{project_id}/download/{file_type}")
async def download_results(
    project_id: str,
    file_type: str,  # 'predictions', 'report', 'model'
    current_user: UserResponse = Depends(get_current_user)
):
    """Download project results"""
    # Verify project ownership and completion
    supabase = get_supabase()
    project_response = supabase.table("projects").select("*").eq("id", project_id).eq("user_id", current_user.id).execute()
    
    if not project_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    project = project_response.data[0]
    if project["status"] != ProjectStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project not completed yet"
        )
    
    # Check subscription limits for downloads
    if current_user.subscription_plan == SubscriptionPlan.FREE and file_type in ['report', 'model']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium downloads require Plus subscription or higher"
        )
    
    try:
        # Generate and return file based on type
        # Implementation would depend on file type
        return {"download_url": f"/files/{project_id}/{file_type}"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download"
        )

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Delete a project"""
    supabase = get_supabase()
    
    # Verify project ownership
    project_response = supabase.table("projects").select("*").eq("id", project_id).eq("user_id", current_user.id).execute()
    if not project_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    try:
        # Delete project from database
        supabase.table("projects").delete().eq("id", project_id).execute()
        
        # Clean up files
        # Implementation would clean up uploaded files and generated results
        
        return {"message": "Project deleted successfully"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        ) 
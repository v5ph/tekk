from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Stripe Configuration
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
    
    # Application Configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    
    # Subscription Limits
    FREE_PLAN_PROJECTS_LIMIT: int = 3
    FREE_PLAN_ROWS_LIMIT: int = 1000
    PLUS_PLAN_PROJECTS_LIMIT: int = 25
    PLUS_PLAN_ROWS_LIMIT: int = 25000
    PRO_PLAN_PROJECTS_LIMIT: int = -1  # Unlimited
    PRO_PLAN_ROWS_LIMIT: int = 100000
    
    class Config:
        env_file = ".env"

settings = Settings() 
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from enum import Enum

class SubscriptionPlan(str, Enum):
    FREE = "free"
    PLUS = "plus"
    PRO = "pro"

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    subscription_plan: SubscriptionPlan = SubscriptionPlan.FREE
    projects_used: int = 0
    created_at: datetime
    updated_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None 
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.auth.utils import verify_token
from app.auth.models import UserResponse, SubscriptionPlan
from app.database import get_supabase
from typing import Optional

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    # Get user from Supabase
    supabase = get_supabase()
    try:
        response = supabase.table("users").select("*").eq("email", token_data.email).execute()
        if not response.data:
            raise credentials_exception
        
        user_data = response.data[0]
        return UserResponse(**user_data)
    except Exception:
        raise credentials_exception

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[UserResponse]:
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

def require_subscription(min_plan: SubscriptionPlan):
    """Dependency factory to require minimum subscription level"""
    async def check_subscription(current_user: UserResponse = Depends(get_current_user)):
        plan_hierarchy = {
            SubscriptionPlan.FREE: 0,
            SubscriptionPlan.PLUS: 1,
            SubscriptionPlan.PRO: 2
        }
        
        if plan_hierarchy[current_user.subscription_plan] < plan_hierarchy[min_plan]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {min_plan.value} subscription or higher"
            )
        return current_user
    
    return check_subscription 
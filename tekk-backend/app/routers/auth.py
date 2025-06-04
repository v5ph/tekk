from fastapi import APIRouter, HTTPException, status, Depends
from app.auth.models import UserCreate, UserLogin, Token, UserResponse, SubscriptionPlan
from app.auth.utils import verify_password, get_password_hash, create_access_token
from app.auth.dependencies import get_current_user
from app.database import get_supabase
from datetime import datetime

router = APIRouter()

@router.post("/signup", response_model=Token)
async def signup(user: UserCreate):
    """Register a new user"""
    supabase = get_supabase()
    
    # Check if user already exists
    existing_user = supabase.table("users").select("*").eq("email", user.email).execute()
    if existing_user.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password and create user
    hashed_password = get_password_hash(user.password)
    now = datetime.utcnow()
    
    user_data = {
        "email": user.email,
        "password_hash": hashed_password,
        "full_name": user.full_name,
        "subscription_plan": SubscriptionPlan.FREE.value,
        "projects_used": 0,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }
    
    try:
        response = supabase.table("users").insert(user_data).execute()
        created_user = response.data[0]
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        user_response = UserResponse(**created_user)
        return Token(access_token=access_token, user=user_response)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Authenticate user and return access token"""
    supabase = get_supabase()
    
    # Get user from database
    response = supabase.table("users").select("*").eq("email", user_credentials.email).execute()
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    user_data = response.data[0]
    
    # Verify password
    if not verify_password(user_credentials.password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user_credentials.email})
    
    user_response = UserResponse(**user_data)
    return Token(access_token=access_token, user=user_response)

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    full_name: str = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update current user information"""
    supabase = get_supabase()
    
    update_data = {"updated_at": datetime.utcnow().isoformat()}
    if full_name is not None:
        update_data["full_name"] = full_name
    
    try:
        response = supabase.table("users").update(update_data).eq("id", current_user.id).execute()
        updated_user = response.data[0]
        return UserResponse(**updated_user)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        ) 
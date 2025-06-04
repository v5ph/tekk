from supabase import create_client, Client
from app.config import settings
import asyncio

supabase: Client = None

async def init_db():
    """Initialize Supabase client"""
    global supabase
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    print("✅ Supabase client initialized")

def get_supabase() -> Client:
    """Get Supabase client instance"""
    if supabase is None:
        raise Exception("Database not initialized. Call init_db() first.")
    return supabase 
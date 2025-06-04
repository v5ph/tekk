from fastapi import APIRouter, HTTPException, status, Depends, Request
from app.auth.dependencies import get_current_user
from app.auth.models import UserResponse, SubscriptionPlan
from app.database import get_supabase
from app.config import settings
import stripe
from datetime import datetime

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Stripe Price IDs (these would be set up in Stripe Dashboard)
PRICE_IDS = {
    SubscriptionPlan.PLUS: "price_plus_monthly",  # Replace with actual Stripe price ID
    SubscriptionPlan.PRO: "price_pro_monthly"     # Replace with actual Stripe price ID
}

@router.get("/subscription")
async def get_subscription_info(current_user: UserResponse = Depends(get_current_user)):
    """Get current user's subscription information"""
    return {
        "current_plan": current_user.subscription_plan,
        "projects_used": current_user.projects_used,
        "limits": get_plan_limits(current_user.subscription_plan)
    }

@router.post("/create-checkout-session")
async def create_checkout_session(
    plan: SubscriptionPlan,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create Stripe checkout session for subscription upgrade"""
    if plan == SubscriptionPlan.FREE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot create checkout session for free plan"
        )
    
    if plan not in PRICE_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subscription plan"
        )
    
    try:
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            payment_method_types=['card'],
            line_items=[{
                'price': PRICE_IDS[plan],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{settings.CORS_ORIGINS[0]}/dashboard?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.CORS_ORIGINS[0]}/pricing",
            metadata={
                'user_id': current_user.id,
                'plan': plan.value
            }
        )
        
        return {"checkout_url": checkout_session.url}
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stripe error: {str(e)}"
        )

@router.post("/create-portal-session")
async def create_portal_session(current_user: UserResponse = Depends(get_current_user)):
    """Create Stripe customer portal session for subscription management"""
    if current_user.subscription_plan == SubscriptionPlan.FREE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription to manage"
        )
    
    try:
        # In a real implementation, you'd store the Stripe customer ID
        # For now, we'll find the customer by email
        customers = stripe.Customer.list(email=current_user.email, limit=1)
        if not customers.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No Stripe customer found"
            )
        
        portal_session = stripe.billing_portal.Session.create(
            customer=customers.data[0].id,
            return_url=f"{settings.CORS_ORIGINS[0]}/dashboard"
        )
        
        return {"portal_url": portal_session.url}
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stripe error: {str(e)}"
        )

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        await handle_checkout_completed(event['data']['object'])
    elif event['type'] == 'customer.subscription.updated':
        await handle_subscription_updated(event['data']['object'])
    elif event['type'] == 'customer.subscription.deleted':
        await handle_subscription_deleted(event['data']['object'])
    
    return {"status": "success"}

async def handle_checkout_completed(session):
    """Handle successful checkout completion"""
    user_id = session['metadata']['user_id']
    plan = session['metadata']['plan']
    
    supabase = get_supabase()
    
    # Update user subscription
    update_data = {
        "subscription_plan": plan,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    supabase.table("users").update(update_data).eq("id", user_id).execute()

async def handle_subscription_updated(subscription):
    """Handle subscription updates"""
    customer_id = subscription['customer']
    
    # Get customer email from Stripe
    customer = stripe.Customer.retrieve(customer_id)
    
    # Determine plan based on subscription
    plan = determine_plan_from_subscription(subscription)
    
    supabase = get_supabase()
    
    # Update user subscription
    update_data = {
        "subscription_plan": plan,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    supabase.table("users").update(update_data).eq("email", customer.email).execute()

async def handle_subscription_deleted(subscription):
    """Handle subscription cancellation"""
    customer_id = subscription['customer']
    
    # Get customer email from Stripe
    customer = stripe.Customer.retrieve(customer_id)
    
    supabase = get_supabase()
    
    # Downgrade to free plan
    update_data = {
        "subscription_plan": SubscriptionPlan.FREE.value,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    supabase.table("users").update(update_data).eq("email", customer.email).execute()

def determine_plan_from_subscription(subscription):
    """Determine subscription plan from Stripe subscription object"""
    # This would map Stripe price IDs to our internal plans
    price_id = subscription['items']['data'][0]['price']['id']
    
    for plan, stripe_price_id in PRICE_IDS.items():
        if price_id == stripe_price_id:
            return plan.value
    
    return SubscriptionPlan.FREE.value

def get_plan_limits(plan: SubscriptionPlan):
    """Get limits for a subscription plan"""
    if plan == SubscriptionPlan.FREE:
        return {
            "projects_per_month": settings.FREE_PLAN_PROJECTS_LIMIT,
            "max_rows": settings.FREE_PLAN_ROWS_LIMIT,
            "features": ["Basic ML models", "CSV export", "Email support"]
        }
    elif plan == SubscriptionPlan.PLUS:
        return {
            "projects_per_month": settings.PLUS_PLAN_PROJECTS_LIMIT,
            "max_rows": settings.PLUS_PLAN_ROWS_LIMIT,
            "features": ["Advanced ML models", "Full reports", "PDF export", "Priority support"]
        }
    elif plan == SubscriptionPlan.PRO:
        return {
            "projects_per_month": "Unlimited",
            "max_rows": settings.PRO_PLAN_ROWS_LIMIT,
            "features": ["All ML models", "Team collaboration", "API access", "Custom models", "Dedicated support"]
        }
    
    return {} 
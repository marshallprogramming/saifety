"""
Stripe billing integration.

Required env vars:
  STRIPE_SECRET_KEY      — Stripe secret key (sk_live_... or sk_test_...)
  STRIPE_WEBHOOK_SECRET  — Signing secret from Stripe dashboard → Webhooks
  STRIPE_PRICE_STARTER   — Price ID for the Starter plan  (price_xxx)
  STRIPE_PRICE_GROWTH    — Price ID for the Growth plan   (price_xxx)

Stripe setup:
  1. Create two recurring products in the Stripe dashboard:
       "Saifety Starter" → $49/month
       "Saifety Growth"  → $199/month
  2. Copy each price ID (price_xxx) into the env vars above.
  3. Create a webhook endpoint pointing to: https://your-domain.com/billing/webhook
     Subscribe to: checkout.session.completed, customer.subscription.deleted

Test locally with the Stripe CLI:
  stripe listen --forward-to localhost:8000/billing/webhook
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from users import User, UserStore

_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
_PRICE_IDS = {
    "starter": os.environ.get("STRIPE_PRICE_STARTER", ""),
    "growth":  os.environ.get("STRIPE_PRICE_GROWTH", ""),
}


def _stripe():
    """Lazy-load Stripe so the app starts cleanly when billing isn't configured."""
    try:
        import stripe as _s
        _s.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        if not _s.api_key:
            raise ValueError("STRIPE_SECRET_KEY is not set")
        return _s
    except ImportError:
        raise RuntimeError("stripe package not installed. Run: pip install stripe")


def create_checkout_session(
    user: "User",
    plan: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout session for a plan upgrade.
    Returns the hosted checkout URL to redirect the user to.
    """
    stripe = _stripe()
    price_id = _PRICE_IDS.get(plan)
    if not price_id:
        raise ValueError(
            f"No Stripe price ID configured for plan '{plan}'. "
            f"Set STRIPE_PRICE_{plan.upper()} env var."
        )

    kwargs = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        # Metadata on both the session and the subscription so the webhook
        # can identify the user regardless of which event type fires.
        "metadata": {"user_id": user.id, "plan": plan},
        "subscription_data": {"metadata": {"user_id": user.id, "plan": plan}},
    }

    # Pre-fill customer info if we have it
    if user.stripe_customer_id:
        kwargs["customer"] = user.stripe_customer_id
    else:
        kwargs["customer_email"] = user.email

    session = stripe.checkout.Session.create(**kwargs)
    return session.url


def create_billing_portal_session(user: "User", return_url: str) -> str:
    """
    Create a Stripe Billing Portal session so the user can manage or cancel
    their subscription without any custom UI on our side.
    """
    stripe = _stripe()
    if not user.stripe_customer_id:
        raise ValueError("User has no Stripe customer — they have not completed checkout yet")
    session = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=return_url,
    )
    return session.url


def handle_webhook(payload_bytes: bytes, sig_header: str, userstore: "UserStore") -> dict:
    """
    Verify and process a Stripe webhook event.

    IMPORTANT: payload_bytes must be the raw request body — NOT parsed JSON.
    Stripe verifies the signature against the exact wire bytes.
    """
    stripe = _stripe()
    if not _WEBHOOK_SECRET:
        raise ValueError("STRIPE_WEBHOOK_SECRET is not configured")

    try:
        event = stripe.Webhook.construct_event(payload_bytes, sig_header, _WEBHOOK_SECRET)
    except Exception:
        raise ValueError("Invalid Stripe webhook signature")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id = data.get("metadata", {}).get("user_id")
        plan    = data.get("metadata", {}).get("plan")
        if user_id and plan:
            userstore.set_plan(user_id, plan)
            userstore.set_stripe_ids(user_id, data.get("customer"), data.get("subscription"))

    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        customer_id = data.get("customer")
        if customer_id:
            user = userstore.get_by_stripe_customer(customer_id)
            if user:
                userstore.set_plan(user.id, "free")

    return {"received": True, "type": event_type}

"""
Simple email sender for transactional messages (password reset etc.).

Configure with environment variables:
    SMTP_HOST   — SMTP server hostname (default: localhost)
    SMTP_PORT   — SMTP port (default: 587)
    SMTP_USER   — SMTP username (optional)
    SMTP_PASS   — SMTP password (optional)
    SMTP_FROM   — From address (default: noreply@saifety.dev)

If SMTP_HOST is not set, emails are logged to stdout instead (dev mode).
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

_SMTP_HOST = os.environ.get("SMTP_HOST")
_SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
_SMTP_USER = os.environ.get("SMTP_USER")
_SMTP_PASS = os.environ.get("SMTP_PASS")
_SMTP_FROM = os.environ.get("SMTP_FROM", "noreply@saifety.dev")


def send_password_reset(to_email: str, reset_url: str) -> None:
    """Send a password reset email. Falls back to stdout in dev mode."""
    subject = "Reset your sAIfety password"
    body_text = f"""Hi,

Someone requested a password reset for your sAIfety account.

Click the link below to choose a new password (expires in 1 hour):

  {reset_url}

If you didn't request this, you can safely ignore this email.

— The sAIfety team
"""
    body_html = f"""<!DOCTYPE html>
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;padding:40px 24px;margin:0">
  <div style="max-width:480px;margin:0 auto;background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:36px">
    <div style="font-size:20px;font-weight:700;margin-bottom:8px">
      <span style="color:#6366f1">s<span style="color:#a78bfa">AI</span>fety</span>
    </div>
    <h2 style="margin:0 0 16px;font-size:18px;font-weight:600">Reset your password</h2>
    <p style="color:#94a3b8;margin:0 0 24px;line-height:1.6">
      Someone requested a password reset for your account.
      Click the button below to choose a new password.
    </p>
    <a href="{reset_url}"
       style="display:inline-block;background:#6366f1;color:#fff;text-decoration:none;
              padding:12px 24px;border-radius:8px;font-weight:600;font-size:14px">
      Reset password
    </a>
    <p style="color:#64748b;font-size:12px;margin:24px 0 0;line-height:1.6">
      This link expires in 1 hour. If you didn't request a reset, ignore this email.
    </p>
  </div>
</body>
</html>"""

    if not _SMTP_HOST:
        # Dev mode — print to console so the developer can click the link
        logger.warning(
            "SMTP_HOST not set — password reset email not sent.\n"
            "Reset URL for %s:\n  %s", to_email, reset_url
        )
        print(f"\n[DEV] Password reset URL for {to_email}:\n  {reset_url}\n")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = _SMTP_FROM
    msg["To"]      = to_email
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as smtp:
            smtp.ehlo()
            if _SMTP_PORT != 25:
                smtp.starttls()
            if _SMTP_USER and _SMTP_PASS:
                smtp.login(_SMTP_USER, _SMTP_PASS)
            smtp.sendmail(_SMTP_FROM, [to_email], msg.as_string())
    except Exception as exc:
        logger.error("Failed to send password reset email to %s: %s", to_email, exc)
        raise

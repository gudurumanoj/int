from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets


NONCE_BYTES = 18
SIGNATURE_BYTES = hashlib.sha256().digest_size


def _claims_bytes(
    *,
    tenant_id: str,
    set_id: str,
    item_id: str,
    position: int,
    user_id: str,
    session_id: str,
    expires_at: str,
    nonce: bytes,
) -> bytes:
    claims = {
        "expires_at": expires_at,
        "item_id": item_id,
        "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
        "position": position,
        "session_id": session_id,
        "set_id": set_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
    }
    return json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_tracking_token(
    *,
    secret: str,
    tenant_id: str,
    set_id: str,
    item_id: str,
    position: int,
    user_id: str,
    session_id: str,
    expires_at: str,
) -> str:
    nonce = secrets.token_bytes(NONCE_BYTES)
    message = _claims_bytes(
        tenant_id=tenant_id,
        set_id=set_id,
        item_id=item_id,
        position=position,
        user_id=user_id,
        session_id=session_id,
        expires_at=expires_at,
        nonce=nonce,
    )
    signature = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(nonce + signature).rstrip(b"=").decode("ascii")


def verify_tracking_token(
    token: str,
    *,
    secret: str,
    tenant_id: str,
    set_id: str,
    item_id: str,
    position: int,
    user_id: str,
    session_id: str,
    expires_at: str,
) -> bool:
    try:
        padding = "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(token + padding)
    except (ValueError, TypeError):
        return False
    if len(decoded) != NONCE_BYTES + SIGNATURE_BYTES:
        return False
    nonce = decoded[:NONCE_BYTES]
    supplied_signature = decoded[NONCE_BYTES:]
    message = _claims_bytes(
        tenant_id=tenant_id,
        set_id=set_id,
        item_id=item_id,
        position=position,
        user_id=user_id,
        session_id=session_id,
        expires_at=expires_at,
        nonce=nonce,
    )
    expected_signature = hmac.new(
        secret.encode("utf-8"), message, hashlib.sha256
    ).digest()
    return hmac.compare_digest(supplied_signature, expected_signature)

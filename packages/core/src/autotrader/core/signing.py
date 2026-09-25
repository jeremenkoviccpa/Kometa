"""Ed25519 signing shared by the risk gate (signs) and execution (verifies).

Lives in core so execution can verify RiskDecisions without importing the
risk package (spec section 4). Keys are raw 32-byte Ed25519 keys, stored hex.
Private keys never live on a server except the risk gate's own decision key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from autotrader.core.hashing import canonical_json
from autotrader.core.models import RiskDecision
from autotrader.core.timeutil import ensure_utc


def generate_keypair() -> tuple[str, str]:
    """(private_hex, public_hex)."""
    k = Ed25519PrivateKey.generate()
    priv = k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
    pub = k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return priv, pub


def load_private(hex_or_path: str | Path) -> Ed25519PrivateKey:
    raw = Path(hex_or_path).read_text().strip() if isinstance(hex_or_path, Path) else hex_or_path
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw))


def load_public(hex_or_path: str | Path) -> Ed25519PublicKey:
    raw = Path(hex_or_path).read_text().strip() if isinstance(hex_or_path, Path) else hex_or_path
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(raw))


def public_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def sign_bytes(key: Ed25519PrivateKey, data: bytes) -> str:
    return key.sign(data).hex()


def verify_bytes(pub: Ed25519PublicKey, data: bytes, signature_hex: str) -> bool:
    try:
        pub.verify(bytes.fromhex(signature_hex), data)
    except (InvalidSignature, ValueError):
        return False
    return True


# ---- risk decisions


def decision_payload(d: RiskDecision) -> bytes:
    body = d.model_dump(mode="json", exclude={"signature"})
    return canonical_json(body).encode()


def sign_decision(key: Ed25519PrivateKey, d: RiskDecision) -> RiskDecision:
    return d.model_copy(update={"signature": sign_bytes(key, decision_payload(d))})


class DecisionRejectedError(Exception):
    pass


@dataclass
class DecisionVerifier:
    """Execution-side check: valid signature, not expired, not replayed, approves the intent."""

    public_key: Ed25519PublicKey
    last_sequence: int = -1

    def verify(self, d: RiskDecision, intent_id: str, now: datetime) -> None:
        if d.signature is None or not verify_bytes(self.public_key, decision_payload(d), d.signature):
            raise DecisionRejectedError("bad or missing risk-gate signature")
        if str(d.intent_id) != intent_id:
            raise DecisionRejectedError("decision is for another intent")
        if ensure_utc(now) > d.expires_at:
            raise DecisionRejectedError("decision expired")
        if d.sequence <= self.last_sequence:
            raise DecisionRejectedError("replayed or out-of-order decision")
        if d.verdict == "reject" or d.approved_lots <= 0:
            raise DecisionRejectedError("decision does not approve an order")
        self.last_sequence = d.sequence

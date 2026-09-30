from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Real-Time Fraud Decisioning Engine",
    version="0.1.0",
)


class TransactionFeatures(BaseModel):
    transaction_id: UUID
    account_id: str
    amount: float = Field(gt=0)
    merchant_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    tx_count_5m: int = Field(default=0, ge=0)
    amount_1h: float = Field(default=0.0, ge=0.0)


class Decision(BaseModel):
    transaction_id: UUID
    risk_score: float
    action: Literal["allow", "review", "block"]
    reason_codes: list[str]
    model_version: str


def bootstrap_score(tx: TransactionFeatures) -> tuple[float, list[str]]:
    """Temporary deterministic scorer used until the trained LightGBM model is wired in."""
    contributions = {
        "HIGH_MERCHANT_RISK": 0.45 * tx.merchant_risk,
        "HIGH_5M_VELOCITY": min(tx.tx_count_5m / 20.0, 1.0) * 0.30,
        "HIGH_1H_AMOUNT": min(tx.amount_1h / 10000.0, 1.0) * 0.15,
        "HIGH_TRANSACTION_AMOUNT": min(tx.amount / 5000.0, 1.0) * 0.10,
    }

    score = min(sum(contributions.values()), 1.0)
    reasons = [
        name
        for name, value in sorted(
            contributions.items(), key=lambda item: item[1], reverse=True
        )
        if value > 0
    ][:3]

    return round(score, 6), reasons


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "fraud-decisioning-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/score", response_model=Decision)
def score_transaction(tx: TransactionFeatures) -> Decision:
    score, reasons = bootstrap_score(tx)

    if score >= 0.75:
        action: Literal["allow", "review", "block"] = "block"
    elif score >= 0.50:
        action = "review"
    else:
        action = "allow"

    return Decision(
        transaction_id=tx.transaction_id,
        risk_score=score,
        action=action,
        reason_codes=reasons,
        model_version="bootstrap-rules-v0",
    )

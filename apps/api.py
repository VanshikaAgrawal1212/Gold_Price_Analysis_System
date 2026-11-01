"""FastAPI inference service for the gold price model.

Start it with:

    py -m uvicorn apps.api:app --reload --port 8000

then open http://127.0.0.1:8000/docs for the generated OpenAPI UI.

This is the right deployment shape when the forecast is consumed by *another
program* (a trading tool, a mobile backend, a batch job). For a human-facing
dashboard use ``apps/streamlit_app.py`` instead.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# allow `uvicorn apps.api:app` from the project root without packaging installs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from configs import settings as S  # noqa: E402
from src import service  # noqa: E402
from src.service import ModelNotTrained  # noqa: E402

app = FastAPI(
    title="Gold Price Analysis & Prediction API",
    version="1.0.0",
    description=(
        "Forecast the forward return / price of gold from macro drivers "
        "(USD/INR, crude, S&P 500, 10Y yield, DXY, silver, CPI)."
    ),
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    model: str | None = Field(
        None,
        description="Trained model name. Omit to use the best-scoring one.",
        examples=["xgboost"],
    )
    horizon: int | None = Field(
        None, ge=1, le=63, description="Business days ahead (informational)."
    )
    refresh_data: bool = Field(
        False, description="Re-download market data instead of using the cache."
    )


class ForecastResponse(BaseModel):
    model: str
    as_of_date: str
    target_date: str
    horizon_days: int
    entry_price_usd: float
    predicted_return: float = Field(description="Forward return, e.g. 0.012 = +1.2%")
    predicted_price_usd: float
    direction: str
    magnitude_pct: float


class HealthResponse(BaseModel):
    status: str
    trained: bool
    best_model: str | None
    trained_at: str | None
    server_time: str


def _to_response(f: dict) -> ForecastResponse:
    ret = float(f["predicted_return"])
    return ForecastResponse(
        model=f["model"],
        as_of_date=f["as_of_date"],
        target_date=f["target_date"],
        horizon_days=f["trained_horizon_days"],
        entry_price_usd=round(float(f["entry_price_usd"]), 2),
        predicted_return=round(ret, 6),
        predicted_price_usd=round(float(f["predicted_price_usd"]), 2),
        direction="UP" if ret > 0 else ("DOWN" if ret < 0 else "FLAT"),
        magnitude_pct=round(abs(ret) * 100, 3),
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus which model is currently loaded (503 if untrained)."""
    try:
        meta = service.load_meta()
        bundle = service.load_bundle()
        best = bundle.get("best_model") or meta.get("best_model")
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return HealthResponse(
        status="ok",
        trained=True,
        best_model=best,
        trained_at=meta.get("trained_at"),
        server_time=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest = ForecastRequest()) -> ForecastResponse:
    """Forecast gold at the trained horizon from the latest available data."""
    try:
        f = service.forecast(
            model_name=req.model,
            horizon=req.horizon,
            refresh_data=req.refresh_data,
        )
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(f)


@app.get("/forecast/{horizon}", response_model=ForecastResponse)
def forecast_get(
    horizon: int,
    model: str | None = Query(None, description="Model name override"),
    refresh: bool = Query(False),
) -> ForecastResponse:
    """GET-friendly variant: ``/forecast/5?model=xgboost``."""
    try:
        f = service.forecast(model_name=model, horizon=horizon, refresh_data=refresh)
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(f)


@app.get("/models")
def models() -> dict:
    try:
        return {"available": service.available_models()}
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict:
    """Held-out test metrics for every trained model."""
    table = service.metrics_table()
    if table is None:
        raise HTTPException(status_code=503, detail="metrics_test.csv missing")
    return {"test_metrics": table.to_dict(orient="records")}


@app.get("/history")
def history(days: int = Query(250, ge=5, le=5000)) -> dict:
    """Recent gold price plus its main macro drivers, as JSON records."""
    tidy = service._feature_frame()
    cols = [S.TARGET_COL, "dxy", "us10y", "usdinr", "silver", "crude_oil"]
    keep = [c for c in cols if c in tidy.columns]
    frame = tidy[keep].tail(days).reset_index()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return {"rows": frame.to_dict(orient="records")}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("apps.api:app", host="127.0.0.1", port=8000, reload=True)

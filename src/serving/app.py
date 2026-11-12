"""FastAPI inference service for location-aware monthly rainfall quantiles (Bengaluru).

Run from the repository root (the model is loaded from the local MLflow registry):
    uvicorn src.serving.app:app --reload

Environment variables:
    RAINFALL_MODEL_URI   default models:/bengaluru_rainfall_by_cell@production
    MLFLOW_TRACKING_URI  default sqlite:///mlflow.db
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.serving.predictor import OutsideServiceArea, PeriodUnavailable, Predictor
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CellInfo,
    Health,
    ModelCard,
    Prediction,
    PredictRequest,
)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Build the app. Pass a ``Predictor`` to skip loading from the registry (tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or Predictor.from_registry()
        yield

    app = FastAPI(
        title="Bengaluru Rainfall Inference API",
        version="0.1.0",
        description=(
            "Monthly rainfall quantiles (Q10/Q50/Q90, mm) for a location in Bengaluru, from the "
            "registered production model. Resolution is the IMD 0.25 degree grid (about 27 km)."
        ),
        lifespan=lifespan,
    )

    def _predictor(request: Request) -> Predictor:
        return request.app.state.predictor

    def _run(request: Request, items: list[PredictRequest]) -> list[Prediction]:
        try:
            return _predictor(request).predict(items)
        except (OutsideServiceArea, PeriodUnavailable) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        p = getattr(request.app.state, "predictor", None)
        return Health(status="ok", model_loaded=p is not None, model_version=p.version if p else None)

    @app.get("/model", response_model=ModelCard)
    def model_card(request: Request) -> ModelCard:
        return _predictor(request).model_card()

    @app.get("/cells", response_model=list[CellInfo])
    def cells(request: Request) -> list[CellInfo]:
        return _predictor(request).cells()

    @app.post("/predict", response_model=Prediction)
    def predict(request: Request, body: PredictRequest) -> Prediction:
        return _run(request, [body])[0]

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(request: Request, body: BatchPredictRequest) -> BatchPredictResponse:
        return BatchPredictResponse(predictions=_run(request, body.items))

    return app


app = create_app()

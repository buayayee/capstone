"""
main.py
-------
FastAPI application entrypoint.
Exposes REST endpoints consumed by the Java backend.

Endpoints:
  GET  /health         - liveness check
  POST /check-document - run fraud classification on document text

Run locally:
    uvicorn api.main:app --reload --port 8000

Docker:
    docker build -t fraud-check-api .
    docker run -p 8000:8000 fraud-check-api
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import load_model, predict
from api.schemas import FraudCheckRequest, FraudCheckResponse, HealthResponse

# ── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the server starts up."""
    load_model()
    yield
    # Cleanup on shutdown (nothing needed here, but hook is available)


app = FastAPI(
    title="Deferment Fraud Check API",
    description="AI-powered document fraud detection for NS deferment applications.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Java backend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Java backend URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Liveness probe — called by Java backend to confirm the service is up."""
    from api.predictor import _classifier
    return HealthResponse(status="ok", model_loaded=_classifier is not None)


@app.post(
    "/check-document",
    response_model=FraudCheckResponse,
    tags=["Fraud Detection"],
    summary="Check if a deferment supporting document is fraudulent",
)
def check_document(request: FraudCheckRequest):
    """
    Accepts extracted document text and returns a fraud verdict.

    Called by the Java backend (be-deferment-nsadmin-defer) during
    the deferment application review workflow.

    Example Java call:
        POST http://localhost:8000/check-document
        {
            "application_id": "DEF-2026-00123",
            "document_text": "...",
            "document_type": "enrollment_letter"
        }
    """
    if not request.document_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="document_text cannot be empty.",
        )

    try:
        result = predict(
            document_text=request.document_text,
            application_id=request.application_id,
        )
        return FraudCheckResponse(**result)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

"""REST service: one resident model, validated input, explanations, audit events."""
from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from src.extraction.predictor import EntityPredictor
from src.explainability.explainer import explain_prediction
from src.guideline_engine.evaluator import GUIDELINE_VERSION, evaluate_guideline, extract_observed_tests

logger = logging.getLogger("clinical_ai.audit")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=12000)
    note_id: StrictStr | None = Field(default=None, min_length=1, max_length=64)
    observed_tests: list[Annotated[StrictStr, Field(min_length=1, max_length=64)]] = Field(default_factory=list, max_length=32)

    @field_validator("text", "note_id")
    @classmethod
    def not_blank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


class StructuredEntities(BaseModel):
    note_id: str
    age: int | None
    sex: str | None
    symptoms: list[str]
    diagnosis: str | None
    medications: list[str]


class AnalyzeResponse(BaseModel):
    request_id: str
    structured_entities: StructuredEntities
    guideline_assessment: dict
    explanation: dict
    warnings: list[str]
    model_version: str
    guideline_version: str
    latency_ms: float


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        engine = predictor or EntityPredictor(model_dir=Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models/entity_extractor"))),
                                              device=os.getenv("MODEL_DEVICE", "cpu"))
        engine.load()
        app.state.predictor = engine
        app.state.lock = Lock()
        logger.info(json.dumps({"event": "model_loaded", "model_version": engine.model_version,
                                "guideline_version": GUIDELINE_VERSION}))
        yield
        app.state.predictor = None

    application = FastAPI(title="Clinical Note Assessment", version="1.0.0", lifespan=lifespan,
                          description="Assessment prototype using the supplied guideline fixture.")

    @application.get("/health")
    def health(request: Request):
        engine = getattr(request.app.state, "predictor", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ok", "model_version": engine.model_version, "guideline_version": GUIDELINE_VERSION}

    @application.post("/analyze_note", response_model=AnalyzeResponse)
    def analyze_note(payload: AnalyzeRequest, request: Request):
        started = perf_counter()
        request_id = str(uuid4())
        engine = getattr(request.app.state, "predictor", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        try:
            with request.app.state.lock:
                prediction = engine.predict({"note_id": payload.note_id or request_id, "text": payload.text})
                evidence = extract_observed_tests(payload.text)
                observed = sorted(set(payload.observed_tests + [item["test"] for item in evidence]))
                structured = prediction["structured"]
                assessment = evaluate_guideline(structured["diagnosis"], structured["medications"], observed)
                explanation = explain_prediction(engine, prediction, assessment)
                explanation["caller_reported_tests"] = payload.observed_tests
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        elapsed = round((perf_counter() - started) * 1000, 3)
        # No note text, entity values, caller note IDs, or attribution text in logs.
        logger.info(json.dumps({"event": "prediction", "request_id": request_id,
                                "model_version": engine.model_version, "guideline_version": GUIDELINE_VERSION,
                                "entity_counts": {k: sum(e["label"] == k for e in prediction["entities"])
                                                  for k in ["age", "sex", "symptoms", "diagnosis", "medications"]},
                                "assessment_status": assessment["status"], "compliant": assessment["compliant"],
                                "warning_count": len(assessment["warnings"]) + len(prediction["warnings"]),
                                "latency_ms": elapsed}))
        return {"request_id": request_id, "structured_entities": {"note_id": prediction["note_id"], **structured},
                "guideline_assessment": assessment, "explanation": explanation,
                "warnings": prediction["warnings"], "model_version": engine.model_version,
                "guideline_version": GUIDELINE_VERSION, "latency_ms": elapsed}

    return application


app = create_app()

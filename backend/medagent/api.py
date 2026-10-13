"""FastAPI application providing clinical REST and SSE endpoints for MedAgent."""

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    BackgroundTasks,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command

from medagent.config import settings
from medagent.schemas import (
    CaseInput,
    DoctorDecision,
    FinalReport,
    CLINICAL_DISCLAIMER,
)
from medagent.graph import (
    graph_app,
    register_event_queue,
    unregister_event_queue,
    get_event_history,
    emit_event,
)
from medagent.audit import get_audit_trail, log_event
from medagent.speech import transcribe_audio

logger = logging.getLogger("medagent.api")

app = FastAPI(
    title="MedAgent API",
    description="Multimodal Multi-Agent Clinical Decision-Support System",
    version="0.1.0",
)

# CORS configuration restricted to local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registry of active cases and states
_cases_state: Dict[str, Dict[str, Any]] = {}

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
ALLOWED_LAB_EXTS = {".txt", ".csv", ".tsv", ".pdf"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filenames to prevent directory traversal."""
    cleaned = Path(filename).name
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)


def run_pipeline_task(case_id: str, case_input: CaseInput) -> None:
    """Background task executing the graph until interruption or completion."""
    config = {"configurable": {"thread_id": case_id}}
    initial_state = {
        "case_input": case_input,
        "plan": None,
        "imaging_report": None,
        "lab_report": None,
        "rag_report": None,
        "clinical_draft": None,
        "safety_report": None,
        "safety_retries": 0,
        "feedback_for_reasoning": None,
        "doctor_decision": None,
        "final_report": None,
        "error": None,
    }
    _cases_state[case_id] = initial_state

    try:
        log_event(case_id, "case_started")
        res = graph_app.invoke(initial_state, config=config)
        _cases_state[case_id].update(res)
    except Exception as e:
        logger.error("Error executing case %s pipeline: %s", case_id, str(e), exc_info=True)
        _cases_state[case_id]["error"] = str(e)
        emit_event(case_id, "error", {"message": f"Graph execution failed: {str(e)}"})
        log_event(case_id, "case_error", details={"error": str(e)})


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "MedAgent",
        "disclaimer": CLINICAL_DISCLAIMER,
    }


@app.get("/api/models/status")
async def models_status() -> Dict[str, Any]:
    """Inspect readiness of domain models, vector backends, and speech models."""
    torch_installed = False
    torch_version = None
    try:
        import torch
        torch_installed = True
        torch_version = torch.__version__
    except ImportError:
        pass

    imaging_ready = False
    imaging_details = "Not loaded"
    try:
        import torchxrayvision as xrv
        imaging_ready = True
        imaging_details = f"TorchXRayVision {xrv.__version__} (DenseNet121 weights available on demand)"
    except ImportError:
        imaging_details = "torchxrayvision not installed"

    whisper_ready = False
    try:
        import faster_whisper
        whisper_ready = True
    except ImportError:
        pass

    return {
        "disclaimer": CLINICAL_DISCLAIMER,
        "torch": {
            "available": torch_installed,
            "version": torch_version,
            "device": "cpu",
        },
        "imaging_model": {
            "model_name": settings.TORCHXRAYVISION_WEIGHTS,
            "ready": imaging_ready,
            "details": imaging_details,
            "medgemma_enabled": settings.ENABLE_MEDGEMMA,
        },
        "llm_engine": {
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL or "default",
            "offline_rule_fallback": True,
        },
        "rag_backend": {
            "backend": settings.VECTOR_BACKEND,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
        "speech": {
            "available": whisper_ready,
            "engine": "faster-whisper" if whisper_ready else "unavailable",
        },
    }


@app.post("/api/cases")
async def create_case(
    background_tasks: BackgroundTasks,
    clinical_text: str = Form(...),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    lab_text: Optional[str] = Form(None),
    files: List[UploadFile] = File(default_factory=list),
) -> Dict[str, str]:
    """Ingest a clinical case with multimodal attachments and trigger LangGraph."""
    case_id = f"case_{uuid.uuid4().hex[:10]}"
    case_upload_dir = settings.UPLOAD_DIR / case_id
    case_upload_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[str] = []
    lab_paths: List[str] = []

    for file in files:
        if not file.filename:
            continue
        safe_name = sanitize_filename(file.filename)
        dest_path = case_upload_dir / safe_name
        ext = dest_path.suffix.lower()

        # Check total size
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} exceeds maximum allowed size of 50 MB.",
            )

        with open(dest_path, "wb") as f:
            f.write(content)

        if ext in ALLOWED_IMAGE_EXTS:
            image_paths.append(str(dest_path))
        elif ext in ALLOWED_LAB_EXTS:
            lab_paths.append(str(dest_path))

    case_input = CaseInput(
        case_id=case_id,
        clinical_text=clinical_text,
        patient_age=patient_age,
        patient_sex=patient_sex if patient_sex in ("male", "female", "other") else None,
        image_paths=image_paths,
        lab_paths=lab_paths,
        lab_text=lab_text or "",
    )

    background_tasks.add_task(run_pipeline_task, case_id, case_input)
    return {"case_id": case_id, "status": "processing"}


@app.get("/api/cases/{case_id}/events")
async def case_events(case_id: str) -> StreamingResponse:
    """Server-Sent Events streaming live pipeline events for the case."""
    queue: asyncio.Queue = asyncio.Queue()
    register_event_queue(case_id, queue)

    async def event_generator():
        # First send historical events
        for past_evt in get_event_history(case_id):
            yield f"event: {past_evt['event']}\ndata: {json.dumps(past_evt)}\n\n"

        try:
            while True:
                evt = await queue.get()
                yield f"event: {evt['event']}\ndata: {json.dumps(evt)}\n\n"
                if evt.get("event") in ("finished", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            unregister_event_queue(case_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str) -> Dict[str, Any]:
    """Retrieve complete state for a case."""
    # Check in-memory state or database checkpoint
    config = {"configurable": {"thread_id": case_id}}
    checkpoint_state = graph_app.get_state(config)
    
    values = checkpoint_state.values if checkpoint_state else {}
    cached = _cases_state.get(case_id, {})
    combined = {**cached, **values}

    if not combined:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    # Helper to serialize models
    def serialize_val(v):
        if hasattr(v, "model_dump"):
            return v.model_dump()
        return v

    serialized = {k: serialize_val(v) for k, v in combined.items()}
    serialized["next_steps"] = list(checkpoint_state.next) if checkpoint_state else []
    serialized["is_interrupted"] = bool(checkpoint_state and checkpoint_state.next and "physician_gate_node" in checkpoint_state.next)
    return serialized


@app.post("/api/cases/{case_id}/decision")
async def submit_decision(case_id: str, decision: DoctorDecision) -> Dict[str, Any]:
    """Submit physician decision to resume interrupted graph from physician gate."""
    config = {"configurable": {"thread_id": case_id}}
    state = graph_app.get_state(config)
    if not state or not state.next:
        # Check if already completed
        if case_id in _cases_state and _cases_state[case_id].get("final_report"):
            return {
                "status": "already_completed",
                "final_report": _cases_state[case_id]["final_report"].model_dump()
                if hasattr(_cases_state[case_id]["final_report"], "model_dump")
                else _cases_state[case_id]["final_report"],
            }
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} is not currently awaiting physician decision.",
        )

    # Resume graph execution using Command(resume=decision)
    try:
        res = graph_app.invoke(Command(resume=decision.model_dump()), config=config)
        _cases_state[case_id].update(res)
        final_rep = res.get("final_report")
        return {
            "status": "success",
            "final_report": final_rep.model_dump() if hasattr(final_rep, "model_dump") else final_rep,
        }
    except Exception as e:
        logger.error("Error resuming case %s with decision: %s", case_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume case: {str(e)}")


@app.get("/api/cases/{case_id}/report.pdf")
async def get_case_pdf(case_id: str) -> FileResponse:
    """Download the finalized ReportLab PDF."""
    pdf_path = settings.REPORTS_DIR / f"report_{case_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report PDF for case {case_id} not found or not released.",
        )
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"MedAgent_Report_{case_id}.pdf",
    )


@app.get("/api/cases/{case_id}/files/{filename}")
async def get_case_file(case_id: str, filename: str) -> FileResponse:
    """Serve case images, heatmaps, and attachments."""
    safe_name = sanitize_filename(filename)
    # Check upload directory
    file_path = settings.UPLOAD_DIR / case_id / safe_name
    if not file_path.exists():
        # Check base data directory or heatmap paths
        file_path = settings.DATA_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=str(file_path))


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribe voice dictation audio file."""
    temp_dir = settings.UPLOAD_DIR / "temp_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"audio_{uuid.uuid4().hex[:8]}_{sanitize_filename(file.filename or 'audio.wav')}"
    try:
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        result = transcribe_audio(str(temp_file))
        return result
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


@app.get("/api/audit")
async def get_audit(limit: int = 100, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve audit events."""
    return get_audit_trail(case_id=case_id, limit=limit)


# Mount static frontend build if it exists
frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

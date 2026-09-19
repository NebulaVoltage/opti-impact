"""FastAPI application adapter for live optical structural monitoring.

Exposes:
  - GET /api/health: Device connection status and frame performance
  - GET /api/video: Live MJPEG stream with Lucas-Kanade overlays
  - GET /api/telemetry: Instantaneous telemetry packet
  - GET /api/spectrum: Instantaneous FFT spectrum and spectrogram history
  - POST /api/reset: Re-centers tracking baseline
  - POST /api/baseline: Captures current response as session baseline
  - GET /api/baseline: Returns captured baseline record
  - WebSocket /ws/optical: Real-time telemetry broadcast channel
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.worker import OpticalPipelineWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("optical_backend.app")

# Global worker instance (single owner of the physical camera)
CAMERA_INDEX = int(os.environ.get("OPTICAL_CAMERA_INDEX", "1"))
optical_worker: Optional[OpticalPipelineWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global optical_worker
    logger.info(f"Starting OpticalPipelineWorker on Camera Index {CAMERA_INDEX}...")
    optical_worker = OpticalPipelineWorker(camera_index=CAMERA_INDEX)
    optical_worker.start()
    yield
    logger.info("Stopping OpticalPipelineWorker...")
    if optical_worker:
        optical_worker.stop()


app = FastAPI(
    title="Opti-Impact Optical Monitoring Backend",
    version="1.0.0",
    description="Live physical optical structural response telemetry and video streaming adapter.",
    lifespan=lifespan,
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def get_health():
    """Check backend and camera hardware status."""
    if optical_worker is None:
        return JSONResponse(status_code=503, content={"status": "initializing", "camera_connected": False})

    return {
        "status": "ok",
        "camera_connected": optical_worker.is_camera_connected,
        "camera_index": optical_worker.camera_index,
        "frame_count": optical_worker.frame_count,
        "connection_error": optical_worker.connection_error,
    }


def mjpeg_frame_generator() -> AsyncGenerator[bytes, None]:
    """Generate MJPEG multipart frame stream from optical pipeline."""
    while True:
        if optical_worker is not None:
            jpeg_bytes = optical_worker.get_latest_jpeg()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
        time.sleep(0.033)


@app.get("/api/video")
async def get_video_stream():
    """Stream live annotated optical video via MJPEG."""
    if optical_worker is None or not optical_worker.is_camera_connected:
        # Return fallback placeholder frame if camera offline
        raise HTTPException(status_code=503, detail="Camera stream currently unavailable.")

    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/telemetry")
async def get_telemetry():
    """Get latest instantaneous telemetry packet."""
    if optical_worker is None:
        raise HTTPException(status_code=503, detail="Worker initializing.")

    data = optical_worker.get_latest_telemetry()
    if data is None:
        return JSONResponse(status_code=204, content={})
    return data


@app.get("/api/spectrum")
async def get_spectrum():
    """Get current FFT spectrum and spectrogram history."""
    if optical_worker is None:
        raise HTTPException(status_code=503, detail="Worker initializing.")

    return {
        "spectrum": optical_worker.get_latest_spectrum() or {},
        "spectrogram": optical_worker.get_spectrogram(),
    }


@app.post("/api/reset")
async def reset_baseline():
    """Re-center feature tracker baseline and motion kinematics."""
    if optical_worker is None:
        raise HTTPException(status_code=503, detail="Worker initializing.")

    optical_worker.reset_baseline()
    return {"status": "ok", "message": "Baseline recentered successfully."}


@app.post("/api/baseline")
async def capture_baseline():
    """Capture current optical response metrics as the baseline."""
    if optical_worker is None:
        raise HTTPException(status_code=503, detail="Worker initializing.")

    record = optical_worker.capture_baseline()
    if record is None:
        raise HTTPException(status_code=400, detail="No telemetry available to capture baseline.")
    return {"status": "ok", "baseline": record}


@app.get("/api/baseline")
async def get_baseline():
    """Retrieve session baseline record."""
    if optical_worker is None:
        raise HTTPException(status_code=503, detail="Worker initializing.")

    return {"baseline": optical_worker.get_baseline()}


# Active WebSocket clients
connected_clients: List[WebSocket] = []
clients_lock = asyncio.Lock()


@app.websocket("/ws/optical")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Real-time optical structural telemetry WebSocket stream."""
    await websocket.accept()
    async with clients_lock:
        connected_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(connected_clients)}")

    loop = asyncio.get_running_loop()

    # Async queue for worker thread to push packets to this websocket
    queue: asyncio.Queue[Dict] = asyncio.Queue(maxsize=10)

    def on_telemetry(packet: Dict):
        try:
            loop.call_soon_threadsafe(
                lambda: queue.put_nowait(packet) if not queue.full() else None
            )
        except Exception:
            pass

    if optical_worker is not None:
        optical_worker.add_subscriber(on_telemetry)

    try:
        while True:
            # Wait for next telemetry frame from acquisition thread
            packet = await queue.get()
            await websocket.send_text(json.dumps(packet))
    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        if optical_worker is not None:
            optical_worker.remove_subscriber(on_telemetry)
        async with clients_lock:
            if websocket in connected_clients:
                connected_clients.remove(websocket)

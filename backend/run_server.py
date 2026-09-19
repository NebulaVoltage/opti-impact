"""CLI launcher for the optical structural monitoring backend server."""

import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Opti-Impact Optical Monitoring Backend Server.")
    parser.add_argument("--camera", type=int, default=1, help="Camera device index (default: 1).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")

    args = parser.parse_args()

    os.environ["OPTICAL_CAMERA_INDEX"] = str(args.camera)

    print("=" * 64)
    print(" OPTI-IMPACT OPTICAL STRUCTURAL MONITORING BACKEND")
    print(f" Camera Index:     {args.camera}")
    print(f" Server URL:       http://{args.host}:{args.port}")
    print(f" Video Stream:     http://{args.host}:{args.port}/api/video")
    print(f" WebSocket Stream: ws://{args.host}:{args.port}/ws/optical")
    print("=" * 64)

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

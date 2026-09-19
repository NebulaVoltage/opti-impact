"""Interactive CLI and live demonstration for physical optical structural motion sensing.

Supports:
  - Device scanning: python -m optical.demo --list-cameras
  - Live optical tracking: python -m optical.demo --camera 1
  - Automated session recording: python -m optical.demo --camera 1 --record --duration 10
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from optical.calibration import PlanarCalibration
from optical.camera import CameraConfig, CameraManager, list_available_cameras
from optical.feature_tracker import OpticalFeatureTracker, TrackerConfig
from optical.optical_motion import OpticalMotionEstimator
from optical.recorder import OpticalSessionRecorder
from optical.visualization import OpticalVisualizer


def parse_roi(roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse 'x,y,w,h' string into integer tuple."""
    if not roi_str:
        return None
    try:
        parts = [int(p.strip()) for p in roi_str.split(",")]
        if len(parts) == 4:
            return (parts[0], parts[1], parts[2], parts[3])
    except Exception:
        pass
    return None


def run_demo(
    camera_index: int = 1,
    width: int = 1280,
    height: int = 720,
    roi: Optional[Tuple[int, int, int, int]] = None,
    record: bool = False,
    duration: Optional[float] = None,
    headless: bool = False,
    session_name: Optional[str] = None,
) -> None:
    """Run live optical structural motion sensing demonstration."""
    print("=" * 64)
    print(" PHYSICAL OPTICAL STRUCTURAL RESPONSE SENSOR")
    print(" MARKER-FREE LUCAS–KANADE OPTICAL FLOW TRACKING")
    print("=" * 64)

    cam_cfg = CameraConfig(camera_index=camera_index, width=width, height=height)
    cam_mgr = CameraManager(config=cam_cfg)

    print(f"Connecting to Camera Index {camera_index}...")
    if not cam_mgr.open():
        print(f"Error: Failed to open camera index {camera_index}.")
        print("Try running 'python -m optical.demo --list-cameras' to inspect available devices.")
        sys.exit(1)

    print(f"Connected: {cam_mgr.actual_width}x{cam_mgr.actual_height} resolution")

    # Default specimen ROI if not provided: central 70% of frame
    if roi is None:
        rx = int(cam_mgr.actual_width * 0.15)
        ry = int(cam_mgr.actual_height * 0.15)
        rw = int(cam_mgr.actual_width * 0.70)
        rh = int(cam_mgr.actual_height * 0.70)
        roi = (rx, ry, rw, rh)

    print(f"Configured Specimen ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    tracker = OpticalFeatureTracker(roi=roi)
    motion_estimator = OpticalMotionEstimator()
    visualizer = OpticalVisualizer()
    calibration = PlanarCalibration()
    recorder = OpticalSessionRecorder(session_name=session_name) if record else None

    window_name = "PHYSICAL OPTICAL STRUCTURAL SENSOR - LUCAS-KANADE"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nTracking active. Controls:")
    print("  [q] Quit session")
    print("  [r] Re-detect features / reset baseline")
    print("  [s] Save frame snapshot")
    print("-" * 64)

    t_start = time.perf_counter()
    latencies = []

    try:
        while True:
            t_frame_start = time.perf_counter()

            # Check duration limit
            if duration is not None and (t_frame_start - t_start) >= duration:
                break

            ret, frame, t_stamp = cam_mgr.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 1. Track features via Lucas-Kanade optical flow
            tracked = tracker.track(frame)

            # 2. Compute robust median kinematics & frequency
            kinematics = motion_estimator.update(tracked, timestamp=t_stamp)

            # 3. Record telemetry if enabled
            if recorder is not None:
                recorder.record_frame(kinematics, camera_fps=cam_mgr.measured_fps)

            t_frame_end = time.perf_counter()
            latencies.append((t_frame_end - t_frame_start) * 1000.0)

            # 4. Render overlay
            annotated = visualizer.draw_frame(
                frame_bgr=frame,
                tracked=tracked,
                kinematics=kinematics,
                camera_fps=cam_mgr.measured_fps,
                roi=roi,
                calibration=calibration,
                show_plot=True,
            )

            # 5. Display GUI if not headless
            if not headless:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    tracker.reset()
                    motion_estimator.reset()
                    print("\n[Tracker Reset] Baseline re-centered and features re-detected.")
                elif key == ord("s"):
                    snap_path = Path("outputs/optical") / f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(str(snap_path), annotated)
                    print(f"\n[Snapshot Saved] {snap_path}")

    finally:
        cam_mgr.release()
        if not headless:
            cv2.destroyAllWindows()

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 64)
    print("OPTICAL SESSION COMPLETE")
    print("=" * 64)
    print(f"Total Session Duration:     {t_total:.2f} s")
    print(f"Total Frames Acquired:      {cam_mgr.frame_count}")
    print(f"Actual Average Frame Rate:  {cam_mgr.measured_fps:.1f} FPS")
    if latencies:
        print(f"Mean Processing Latency:    {np.mean(latencies):.2f} ms")
        print(f"Median Latency:             {np.median(latencies):.2f} ms")
        print(f"95th Percentile Latency:    {np.percentile(latencies, 95):.2f} ms")
        print(f"Max Processing Latency:     {np.max(latencies):.2f} ms")

    if recorder is not None:
        csv_file = recorder.save()
        print(f"Saved Session Telemetry:    {csv_file} ({recorder.record_count} records)")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="Real physical optical structural motion sensing CLI.")
    parser.add_argument("--list-cameras", action="store_true", help="Probe and list available video cameras.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index to open (default: 1 for external USB).")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720).")
    parser.add_argument("--roi", type=str, default=None, help="Specimen ROI as 'x,y,w,h'.")
    parser.add_argument("--record", action="store_true", help="Record session data to CSV.")
    parser.add_argument("--duration", type=float, default=None, help="Session duration in seconds before auto-exit.")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window (for batch/testing).")
    parser.add_argument("--session-name", type=str, default=None, help="Custom filename prefix for recording.")

    args = parser.parse_args()

    if args.list_cameras:
        print("Scanning connected video capture devices...")
        cams = list_available_cameras()
        print("\nDiscovered Devices:")
        for c in cams:
            status = "AVAILABLE" if c.is_available else "NOT OPENED"
            print(f"  Camera Index {c.index}: {status} | Resolution: {c.width}x{c.height} | Measured FPS: {c.actual_fps:.1f}")
        return

    run_demo(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        roi=parse_roi(args.roi),
        record=args.record,
        duration=args.duration,
        headless=args.headless,
        session_name=args.session_name,
    )


if __name__ == "__main__":
    main()

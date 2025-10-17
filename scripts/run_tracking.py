import argparse
from src.tracking.yolo_bytetrack import run_tracking

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="tracked_yolo_bytetrack.mp4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_tracking(args.input, args.output, device=args.device)

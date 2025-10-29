INPUT  = r"영상경로"
OUTPUT = "트래킹 영상"

MODEL  = "yolov8m.pt"
DEVICE = "cpu"        
CLASSES = [0, 32] 
CONF = 0.25
MAX_W = 1280
SHOW_LABEL = True

import cv2, numpy as np, os, math
from ultralytics import YOLO

model = YOLO(MODEL)

cap = cv2.VideoCapture(INPUT)
assert cap.isOpened(), f"영상 열기 실패: {INPUT}"
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

scale = 1.0
if MAX_W and W > MAX_W:
    scale = MAX_W / W
    W = int(W * scale); H = int(H * scale)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (W, H))

def color_for(cls_id):
    # 사람=초록, 공=오렌지
    return (0,255,0) if cls_id == 0 else (0,180,255)

frame_idx = 0
for result in model.track(
        source=INPUT, 
        conf=CONF,
        iou=0.45,
        classes=CLASSES,
        device=DEVICE,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True):

    frame = result.orig_img
    if scale != 1.0:
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    vis = frame.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids   = result.boxes.id
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        if ids is not None:
            ids = ids.int().cpu().numpy()
        else:
            ids = np.array([-1]*len(boxes))

        for (x1,y1,x2,y2), tid, c in zip(boxes, ids, clss):
            if scale != 1.0:
                x1,y1,x2,y2 = x1*scale, y1*scale, x2*scale, y2*scale
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])

            col = color_for(c)
            cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)

            if SHOW_LABEL:
                label = f"ID {tid} {'player' if c==0 else 'ball'}"
                cv2.putText(vis, label, (x1, max(0, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    writer.write(vis)
    frame_idx += 1

writer.release()
print(OUTPUT)

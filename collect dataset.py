import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime

# ===== CONFIG =====
CAMERA_INDEX      = 1
LINE_START        = (300, 280)
LINE_END          = (800, 280)
NUM_SAMPLES       = 50
OBJECT_THRESHOLD  = 40   # diff vs background → object present
STABLE_FRAMES     = 8    # frames object must be on line to trigger
CAPTURE_DELAY_SEC = 1.0  # wait after trigger before capturing
CAPTURE_COUNT     = 10   # number of pictures to take
COOLDOWN_SEC      = 3.0
CAM_W, CAM_H      = 1280, 720
BASE_SAVE_FOLDER  = r"C:\Users\phassachol.jariya\Desktop\dataset"
# ==================

os.makedirs(BASE_SAVE_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_SAVE_FOLDER, "log.txt")),
        logging.StreamHandler()
    ]
)

def sample_line(frame, p1, p2, n):
    xs = np.linspace(p1[0], p2[0], n).astype(int)
    ys = np.linspace(p1[1], p2[1], n).astype(int)
    return np.mean([frame[y, x] for x, y in zip(xs, ys)], axis=0)

def diff(a, b):
    return np.linalg.norm(np.array(a, float) - np.array(b, float))

def save_frame(frame, index=None):
    folder = os.path.join(BASE_SAVE_FOLDER, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(folder, exist_ok=True)
    suffix = f"_{index:02d}" if index is not None else ""
    path = os.path.join(folder, f"capture_{datetime.now().strftime('%H%M%S_%f')}{suffix}.png")
    if cv2.imwrite(path, frame):
        logging.info(f"[SAVED] {path}")
    else:
        logging.error("[ERROR] Save failed")

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        logging.error("Cannot open webcam")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error("Cannot read first frame")
        return

    frame = cv2.flip(frame, 1)
    bg_color       = sample_line(frame, LINE_START, LINE_END, NUM_SAMPLES)
    state          = "EMPTY"
    stable_counter = 0
    trigger_time   = 0
    last_capture   = 0

    logging.info("Ready — 'b' = update background, 'q' = quit")

    STATE_COLORS = {
        "EMPTY":    (100, 100, 100),
        "WAITING":  (0, 165, 255),
        "DELAY":    (0, 255, 255),
        "COOLDOWN": (255, 0, 0),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame read failed")
            continue

        frame = cv2.flip(frame, 1)
        cur = sample_line(frame, LINE_START, LINE_END, NUM_SAMPLES)
        obj_present = diff(cur, bg_color) > OBJECT_THRESHOLD

        if state == "EMPTY":
            if obj_present:
                state = "WAITING"
                stable_counter = 0
                logging.info("Object on line")

        elif state == "WAITING":
            if not obj_present:
                state = "EMPTY"
                stable_counter = 0
                logging.info("Object left")
            else:
                stable_counter += 1
                logging.info(f"Holding [{stable_counter}/{STABLE_FRAMES}]")
                if stable_counter >= STABLE_FRAMES:
                    trigger_time = time.time()
                    state = "DELAY"
                    logging.info(f"Triggered — waiting {CAPTURE_DELAY_SEC}s")

        elif state == "DELAY":
            if time.time() - trigger_time >= CAPTURE_DELAY_SEC:
                if time.time() - last_capture > COOLDOWN_SEC:
                    logging.info(f">>> BURST CAPTURE x{CAPTURE_COUNT} <<<")
                    for i in range(CAPTURE_COUNT):
                        ret_b, burst = cap.read()
                        if ret_b:
                            burst = cv2.flip(burst, 1)
                            save_frame(burst, index=i + 1)
                    last_capture = time.time()
                state = "COOLDOWN"

        elif state == "COOLDOWN":
            if not obj_present:
                state = "EMPTY"
                logging.info("Ready for next object")

        color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.line(frame, LINE_START, LINE_END, color, 2)
        cv2.putText(frame, f"{state}  bg_diff:{diff(cur, bg_color):.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Conveyor Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            bg_color = sample_line(frame, LINE_START, LINE_END, NUM_SAMPLES)
            logging.info("Background updated")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("=== END ===")

if __name__ == "__main__":
    main()

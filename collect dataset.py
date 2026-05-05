import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime

# ===== CONFIG =====
CAMERA_INDEX    = 0
LINE_START      = (100, 280)
LINE_END        = (800, 280)
NUM_SAMPLES     = 50
OBJECT_THRESHOLD = 40   # diff vs background → object present
STABLE_THRESHOLD = 10   # diff vs prev frame → object stopped
STABLE_FRAMES   = 8
COOLDOWN_SEC    = 2.0
CAM_W, CAM_H    = 1280, 720
BASE_SAVE_FOLDER = r"C:\Users\phassachol.jariya\Desktop\dataset"
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

def save_frame(frame):
    folder = os.path.join(BASE_SAVE_FOLDER, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"capture_{datetime.now().strftime('%H%M%S_%f')}.png")
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
    bg_color   = sample_line(frame, LINE_START, LINE_END, NUM_SAMPLES)
    prev_color = bg_color.copy()
    state          = "EMPTY"
    stable_counter = 0
    last_capture   = 0

    logging.info("Ready — 'b' = update background, 'q' = quit")

    STATE_COLORS = {
        "EMPTY":    (100, 100, 100),
        "MOVING":   (0, 165, 255),
        "STABLE":   (0, 255, 0),
        "COOLDOWN": (255, 0, 0),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame read failed")
            continue

        frame = cv2.flip(frame, 1)
        cur = sample_line(frame, LINE_START, LINE_END, NUM_SAMPLES)

        obj_present = diff(cur, bg_color)   > OBJECT_THRESHOLD
        obj_moving  = diff(cur, prev_color) > STABLE_THRESHOLD

        if state == "EMPTY":
            if obj_present:
                state = "MOVING"
                stable_counter = 0
                logging.info("Object detected")

        elif state == "MOVING":
            if not obj_present:
                state = "EMPTY"
                logging.info("Object left")
            elif not obj_moving:
                stable_counter += 1
                logging.info(f"Stable [{stable_counter}/{STABLE_FRAMES}]")
                if stable_counter >= STABLE_FRAMES:
                    state = "STABLE"
            else:
                stable_counter = 0

        elif state == "STABLE":
            if time.time() - last_capture > COOLDOWN_SEC:
                logging.info(">>> CAPTURE <<<")
                save_frame(frame)
                last_capture = time.time()
            state = "COOLDOWN"

        elif state == "COOLDOWN":
            if not obj_present:
                state = "EMPTY"
                logging.info("Ready for next object")

        prev_color = cur

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

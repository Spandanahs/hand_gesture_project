"""
Neon Fiber Hands — two-hand tracking with glowing nodes and fiber-optic
strands connecting matching fingertips between both hands.

Run:
    pip install opencv-python mediapipe numpy
    python handtracking.py

Quit with 'q' or Esc.
"""

import math
import random
import time
from collections import deque
from turtle import distance

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

FINGER_TIPS = [4, 8, 12, 16, 20]
NUM_LANDMARKS = 21

TOTAL_STRANDS = 180
STRANDS_PER_FINGER = TOTAL_STRANDS // len(FINGER_TIPS)
STRAND_SEGMENTS = 8
STRAND_JITTER_RATIO = 0.10
STRAND_SHIMMER_SPEED = 1.6

TRAIL_LENGTH = 16

NODE_CORE_RADIUS = 3
NODE_GLOW_RADIUS = 7

BLOOM_DOWNSCALE = 4
BLOOM_KERNEL = 25
BLOOM_PASSES = 2
BLOOM_BOOST = 1.6

SPEED_NORM = 900.0
MAX_PREV_GAP = 0.4

COLOR_CYCLE_SPEED = 0.05
NEON_COLORS = [
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
]

mp_hands = mp.solutions.hands


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def get_cycle_color(t):
    n = len(NEON_COLORS)
    pos = (t * COLOR_CYCLE_SPEED) % n
    idx = int(pos)
    frac = pos - idx
    return lerp_color(NEON_COLORS[idx], NEON_COLORS[(idx + 1) % n], frac)


def scale_color(color, factor):
    return tuple(min(255, max(0, int(c * factor))) for c in color)


def bezier_points(p0, p1, perp_unit, offset, n=STRAND_SEGMENTS):
    tt = np.linspace(0.0, 1.0, n, dtype=np.float32)
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    mid = (p0 + p1) / 2.0 + perp_unit * offset
    a = ((1 - tt) ** 2)[:, None]
    b = (2 * (1 - tt) * tt)[:, None]
    c = (tt ** 2)[:, None]
    pts = a * p0 + b * mid + c * p1
    return pts.reshape((-1, 1, 2)).astype(np.int32)


def render_bloom(base, glow_source):
    h, w = glow_source.shape[:2]
    sw, sh = max(1, w // BLOOM_DOWNSCALE), max(1, h // BLOOM_DOWNSCALE)
    small = cv2.resize(glow_source, (sw, sh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    k = BLOOM_KERNEL if BLOOM_KERNEL % 2 == 1 else BLOOM_KERNEL + 1
    for _ in range(BLOOM_PASSES):
        small = cv2.GaussianBlur(small, (k, k), 0)
    bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    bloom = np.clip(bloom * BLOOM_BOOST, 0, 255).astype(np.uint8)
    result = cv2.add(base, bloom)
    result = cv2.add(result, (glow_source // 2))
    return result


class FingerState:
    def __init__(self):
        self.prev_pos = None
        self.prev_time = None
        self.speed = 0.0
        self.trail = deque(maxlen=TRAIL_LENGTH)


class HandState:
    def __init__(self):
        self.fingers = {tip: FingerState() for tip in FINGER_TIPS}


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    rng = random.Random(7)
    jitter_pattern = [
        [rng.uniform(-1.0, 1.0) for _ in range(STRANDS_PER_FINGER)]
        for _ in FINGER_TIPS
    ]
    shimmer_phase = [
        [rng.uniform(0, 2 * math.pi) for _ in range(STRANDS_PER_FINGER)]
        for _ in FINGER_TIPS
    ]

    hand_states = [HandState(), HandState()]
    start_time = time.time()

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
        

            if results.multi_hand_landmarks:
               print("Hand detected")
            canvas = frame.copy()
            glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

            now = time.time()
            t = now - start_time
            base_color = get_cycle_color(t)

            hands_pts = []

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    state = hand_states[hand_idx]
                    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_landmarks.landmark)}
                    hands_pts.append(pts)
                    # Virtual Mouse Control
                    index_x, index_y = pts[8]

                    screen_w, screen_h = pyautogui.size()
  
                    mouse_x = np.interp(index_x, [0, w], [0, screen_w])
                    mouse_y = np.interp(index_y, [0, h], [0, screen_h])

                    pyautogui.moveTo(mouse_x, mouse_y) 
                    thumb_x, thumb_y = pts[4]

                    distance = ((index_x - thumb_x)**2 + (index_y - thumb_y)**2) ** 0.5

                    if distance < 80:
                     print("Click detected!")
                     pyautogui.click()
                     time.sleep(0.5)
                    # Virtual Mouse Control
                    index_x, index_y = pts[8]

                    screen_w, screen_h = pyautogui.size()

                    mouse_x = np.interp(index_x, [0, w], [0, screen_w])
                    mouse_y = np.interp(index_y, [0, h], [0, screen_h])

                    pyautogui.moveTo(mouse_x, mouse_y)

                    for lm_idx in range(NUM_LANDMARKS):
                        p = pts[lm_idx]
                        cv2.circle(canvas, p, NODE_CORE_RADIUS, base_color, -1, cv2.LINE_AA)
                        cv2.circle(glow_layer, p, NODE_GLOW_RADIUS, base_color, -1, cv2.LINE_AA)

                    for tip in FINGER_TIPS:
                        pos = pts[tip]
                        fstate = state.fingers[tip]

                        if fstate.prev_pos is not None and fstate.prev_time is not None:
                            dt = now - fstate.prev_time
                            if 0 < dt <= MAX_PREV_GAP:
                                dist = math.hypot(pos[0] - fstate.prev_pos[0], pos[1] - fstate.prev_pos[1])
                                fstate.speed = dist / dt
                            else:
                                fstate.speed = 0.0
                        fstate.prev_pos = pos
                        fstate.prev_time = now

                        speed_norm = min(fstate.speed / SPEED_NORM, 1.6)
                        brightness = 0.5 + speed_norm
                        fstate.trail.appendleft((pos, brightness))

                        for j, (tpos, tb) in enumerate(fstate.trail):
                            fade = 1.0 - (j / TRAIL_LENGTH)
                            if fade <= 0:
                                continue
                            radius = max(1, int(4 * fade))
                            pcolor = scale_color(base_color, tb * fade)
                            cv2.circle(glow_layer, tpos, radius, pcolor, -1, cv2.LINE_AA)

            if len(hands_pts) == 2:
                pts_a, pts_b = hands_pts[0], hands_pts[1]
                for f_idx, tip in enumerate(FINGER_TIPS):
                    p0 = pts_a[tip]
                    p1 = pts_b[tip]
                    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                    dist = max(1.0, math.hypot(dx, dy))
                    perp_unit = np.array([-dy / dist, dx / dist], dtype=np.float32)
                    max_offset = dist * STRAND_JITTER_RATIO

                    for k in range(STRANDS_PER_FINGER):
                        shimmer = 0.6 + 0.4 * math.sin(t * STRAND_SHIMMER_SPEED + shimmer_phase[f_idx][k])
                        offset = jitter_pattern[f_idx][k] * max_offset * shimmer
                        pts = bezier_points(p0, p1, perp_unit, offset)
                        strand_color = scale_color(base_color, 0.55 + 0.45 * shimmer)
                        cv2.polylines(canvas, [pts], False, strand_color, 1, cv2.LINE_AA)
                        cv2.polylines(glow_layer, [pts], False, strand_color, 2, cv2.LINE_AA)

            final = render_bloom(canvas, glow_layer)

            cv2.imshow("Neon Fiber Hands", final)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
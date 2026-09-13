#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared landmark extraction and normalization utilities (MediaPipe Tasks API).

Compatible with mediapipe >= 0.10 (Tasks-based API).

The new API returns landmarks already normalized to [0.0, 1.0] relative to the
image dimensions, so no pixel-coordinate conversion is needed.

Feature dimensions:
  Static model : STATIC_DIM = 42  values
                 (21 landmarks × [x, y], wrist-relative, max-normalized)
"""

import itertools

# ── Feature dimensions ────────────────────────────────────────────────────────
STATIC_DIM = 21 * 2   # 42  (x, y only, relative + normalised)


# ── Static extraction ─────────────────────────────────────────────────────────

def extract_static_row(hand_landmarks) -> list[float]:
    """
    Extract one hand's 21 landmarks → 42-dim normalised vector.

    Args:
        hand_landmarks: a single element from HandLandmarkerResult.hand_landmarks
                        (list of 21 NormalizedLandmark objects with .x and .y
                        already in [0, 1] image-space).

    Returns:
        list of 42 floats, all in [-1, 1].

    Process:
      1. Use the normalised (x, y) coordinates directly (no pixel conversion).
      2. Make relative to the wrist (landmark index 0).
      3. Flatten to a 1-D list.
      4. Normalise by max absolute value so all values ∈ [-1, 1].
    """
    pts = [[lm.x, lm.y] for lm in hand_landmarks]

    # Relative to wrist
    bx, by = pts[0]
    for pt in pts:
        pt[0] -= bx
        pt[1] -= by

    flat = list(itertools.chain.from_iterable(pts))
    max_val = max(map(abs, flat)) or 1.0
    return [v / max_val for v in flat]

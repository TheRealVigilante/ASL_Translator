#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest a Kaggle-style ASL image dataset (folder-per-class structure) and
extract MediaPipe hand landmarks → static_data.csv.

Uses the MediaPipe Tasks API (mediapipe >= 0.10).
Requires: model/hand_landmarker.task  (downloaded automatically if absent)

Expected source structure:
    source_dir/
        A/  image1.jpg  image2.jpg  ...
        B/  ...
        Z/  ...
        space/  ...

Usage (from asl-classifier/ directory):
    python -m ingest.from_images \\
        --source ../asl-alphabet/asl_alphabet_train/asl_alphabet_train \\
        --out_csv   model/static_classifier/static_data.csv \\
        --out_labels model/static_classifier/static_labels.csv
"""

import argparse
import csv
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.landmark_utils import extract_static_row

# ── Default model path ────────────────────────────────────────────────────────
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model', 'hand_landmarker.task',
)
_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
)


def _ensure_model(model_path: str) -> None:
    """Download the hand landmarker model bundle if not present."""
    if os.path.exists(model_path):
        return
    print(f'[ingest] Model not found at {model_path}')
    print(f'[ingest] Downloading from {_MODEL_URL} ...')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    print(f'[ingest] Downloaded → {model_path}')


# ─────────────────────────────────────────────────────────────────────────────

def process_image_dataset(
    source_dir: str,
    out_csv: str,
    out_labels: str,
    model_path: str = _DEFAULT_MODEL,
    min_confidence: float = 0.5,
    skip_classes: list[str] | None = None,
) -> None:
    """
    Walk a folder-per-class image dataset, extract one-hand landmarks per image,
    and write rows to a CSV.

    Skips images where MediaPipe detects no hand.
    """
    skip_classes = set(skip_classes or [])
    _ensure_model(model_path)

    # ── Build HandLandmarker (IMAGE mode — processes one image at a time) ─────
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    class_names = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
        and d not in skip_classes
    ])

    if not class_names:
        print(f'[ERROR] No subdirectories found in: {source_dir}')
        landmarker.close()
        return

    print(f'Found {len(class_names)} classes: {class_names}')

    # Write labels
    os.makedirs(os.path.dirname(out_labels) or '.', exist_ok=True)
    with open(out_labels, 'w', encoding='utf-8-sig') as f:
        for name in class_names:
            f.write(name + '\n')

    processed = skipped = 0
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for class_idx, class_name in enumerate(
                tqdm(class_names, desc='Processing classes')):
            class_dir = os.path.join(source_dir, class_name)
            files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for img_file in tqdm(files, desc=f'  {class_name}', leave=False):
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    skipped += 1
                    continue

                # Convert BGR → RGB and wrap in mp.Image
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    row = extract_static_row(result.hand_landmarks[0])
                    writer.writerow([class_idx, *row])
                    processed += 1
                else:
                    skipped += 1

    landmarker.close()

    print(f'\n✅ Static dataset extraction complete.')
    print(f'   Rows written : {processed:,}')
    print(f'   Skipped      : {skipped:,}  (no hand detected)')
    print(f'   Output CSV   : {out_csv}')
    print(f'   Labels       : {out_labels}')


# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True,
                   help='Root folder of the image dataset (one sub-folder per class)')
    p.add_argument('--out_csv',
                   default='model/static_classifier/static_data.csv',
                   help='Output CSV path for landmark data')
    p.add_argument('--out_labels',
                   default='model/static_classifier/static_labels.csv',
                   help='Output CSV with one class name per line')
    p.add_argument('--model',
                   default=_DEFAULT_MODEL,
                   help='Path to hand_landmarker.task model bundle')
    p.add_argument('--min_confidence', type=float, default=0.5,
                   help='MediaPipe min detection confidence (default: 0.5)')
    p.add_argument('--skip', nargs='*', default=[],
                   metavar='CLASS',
                   help='Class folder names to skip (e.g. --skip nothing delete)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    process_image_dataset(
        source_dir=args.source,
        out_csv=args.out_csv,
        out_labels=args.out_labels,
        model_path=args.model,
        min_confidence=args.min_confidence,
        skip_classes=args.skip,
    )

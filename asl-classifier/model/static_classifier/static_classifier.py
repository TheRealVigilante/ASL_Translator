#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFLite wrapper for the static (single-frame) ASL letter classifier.

Architecture expects:
    Input  : (1, 42)  — 21 hand landmarks × [x, y], wrist-relative, normalised
    Output : (1, N)   — class probabilities (N = number of label classes)

Pattern reused from the original project's KeyPointClassifier.
"""
import os

import numpy as np

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class StaticClassifier:
    """
    Wraps a TFLite single-frame hand sign classifier.
    Gracefully handles the case where the .tflite model does not exist yet
    (returns -1, 0.0 until trained and exported).
    """

    def __init__(
        self,
        model_path: str = 'model/static_classifier/static_classifier.tflite',
        num_threads: int = 2,
        score_threshold: float = 0.60,
    ):
        self.score_threshold = score_threshold
        self._ready = False

        if not _TF_AVAILABLE:
            print('[StaticClassifier] TensorFlow not installed.')
            return

        if not os.path.exists(model_path):
            print(f'[StaticClassifier] Model not found at: {model_path}')
            print('  → Run train_static.ipynb to train and export the model.')
            return

        try:
            self._interp = tf.lite.Interpreter(
                model_path=model_path, num_threads=num_threads)
            self._interp.allocate_tensors()
            self._in  = self._interp.get_input_details()
            self._out = self._interp.get_output_details()
            self._ready = True
            print(f'[StaticClassifier] Loaded: {model_path}')
        except Exception as exc:
            print(f'[StaticClassifier] Failed to load model: {exc}')

    @property
    def ready(self) -> bool:
        return self._ready

    def __call__(self,
                 landmark_list: 'list[float] | np.ndarray') -> tuple[int, float]:
        """
        Classify a single-frame hand landmark vector.

        Args:
            landmark_list: 42-element list or numpy array

        Returns:
            (class_index, confidence)
            Returns (-1, 0.0) if model not ready or confidence < threshold.
        """
        if not self._ready:
            return -1, 0.0

        inp = np.array([landmark_list], dtype=np.float32)
        self._interp.set_tensor(self._in[0]['index'], inp)
        self._interp.invoke()

        probs = np.squeeze(
            self._interp.get_tensor(self._out[0]['index']))

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])

        return (idx, conf) if conf >= self.score_threshold else (-1, conf)

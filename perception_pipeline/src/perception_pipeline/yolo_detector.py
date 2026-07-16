from dataclasses import dataclass
from typing import List, Dict, Any
from threading import Lock
from time import perf_counter
import os

import logging
import cv2
import torch
import numpy as np

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    track_id: int = -1


@dataclass
class DetectionOutput:
    latency_ms: float
    detections: List[Detection]
    image_shape: tuple


class BaseDetector:
    def detect(self, image):
        raise NotImplementedError


class YoloDetector(BaseDetector):

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config or {}

        # Load configuration params
        self.model_path = self.cfg.get("model_path")
        if self.model_path and not os.path.isabs(self.model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, self.model_path)

        self.device = self.cfg.get(
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.img_size = self.cfg.get("image_size", 640)
        self.conf_thresh = self.cfg.get("confidence_threshold", 0.25)
        self.iou_thresh = self.cfg.get("iou_threshold", 0.45)

        self.fp16 = self.cfg.get("fp16", True) and "cuda" in self.device
        self.allowed_classes = self.cfg.get("allowed_classes", [])
        self.roi_cfg = self.cfg.get("roi", {})
        self.enable_tracking = self.cfg.get("tracking", False)

        perf_cfg = self.cfg.get("performance", {})
        self.batch_size = perf_cfg.get("batch_size", 1)
        self.warmup_runs = perf_cfg.get("warmup_runs", 3)

        # Track runtime performance metrics
        self.frame_count = 0
        self.total_latency = 0.0
        self.last_latency = 0.0
        self.max_latency = 0.0

        self.lock = Lock()
        torch.backends.cudnn.benchmark = True

        self.model = None
        self.class_names = {}

        self._load_model()
        self._warmup()

    def _load_model(self):
        logger.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.fuse()
        self.model.to(self.device)
        self.class_names = self.model.names
        logger.info(f"YOLO loaded on: {self.device}")

    def _warmup(self):
        logger.info(f"Running {self.warmup_runs} warmup runs...")
        dummy = np.zeros(
            (self.img_size, self.img_size, 3),
            dtype=np.uint8
        )
        for _ in range(self.warmup_runs):
            self.detect(dummy, warmup=True)
        logger.info("Warmup complete")

    def _validate_image(self, image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be numpy array")

        if image.ndim not in (2, 3):
            raise ValueError("Invalid image dimensions")

        if image.size == 0:
            raise ValueError("Empty image")

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError("Invalid image channels")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)

        # Handle different channel layouts
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return self.apply_roi(image)

    def apply_roi(self, image: np.ndarray) -> np.ndarray:
        if not self.roi_cfg.get("enable", False):
            return image

        h, w = image.shape[:2]
        x_min = max(0, self.roi_cfg.get("x_min", 0))
        y_min = max(0, self.roi_cfg.get("y_min", 0))
        x_max = min(w, self.roi_cfg.get("x_max", w))
        y_max = min(h, self.roi_cfg.get("y_max", h))

        roi = image[y_min:y_max, x_min:x_max]
        return roi if roi.size > 0 else image

    @torch.no_grad()
    def infer(self, images):
        if not self.lock.acquire(timeout=1.0):
            raise RuntimeError("Inference lock timeout")

        try:
            common_args = dict(
                source=images,
                imgsz=self.img_size,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                half=self.fp16,
                verbose=False
            )

            if self.enable_tracking:
                results = self.model.track(
                    persist=True,
                    tracker="bytetrack.yaml",
                    **common_args
                )
            else:
                results = self.model.predict(**common_args)
        finally:
            self.lock.release()

        return results

    def postprocess(self, result, orig_shape=None) -> List[Detection]:
        detections = []
        if result.boxes is None:
            return detections

        x_offset = 0
        y_offset = 0
        if self.roi_cfg.get("enable", False) and orig_shape is not None:
            x_offset = max(0, self.roi_cfg.get("x_min", 0))
            y_offset = max(0, self.roi_cfg.get("y_min", 0))

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, "unknown")

            # Filter classes based on configuration
            if self.allowed_classes:
                if class_name not in self.allowed_classes:
                    continue

            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0])

            xyxy = box.xyxy[0].detach().cpu().numpy().copy()
            
            # Re-apply ROI coordinates offset if needed
            if x_offset > 0 or y_offset > 0:
                xyxy[0] += x_offset
                xyxy[2] += x_offset
                xyxy[1] += y_offset
                xyxy[3] += y_offset

            confidence = float(box.conf[0])

            detections.append(
                Detection(
                    bbox=xyxy,
                    confidence=confidence,
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id
                )
            )

        return detections

    def detect(self, image: np.ndarray, warmup: bool = False) -> DetectionOutput:
        return self.detect_batch([image], warmup=warmup)[0]

    def detect_batch(self, images: List[np.ndarray], warmup: bool = False) -> List[DetectionOutput]:
        processed_images = [self.preprocess(img) for img in images]

        if "cuda" in self.device:
            torch.cuda.synchronize()

        start = perf_counter()
        try:
            results = self.infer(processed_images)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [
                DetectionOutput(
                    latency_ms=0.0,
                    detections=[],
                    image_shape=img_orig.shape
                )
                for img_orig in images
            ]

        if "cuda" in self.device:
            torch.cuda.synchronize()

        latency_ms = (perf_counter() - start) * 1000.0
        outputs = []

        for img_orig, result in zip(images, results):
            detections = self.postprocess(result, img_orig.shape)
            outputs.append(
                DetectionOutput(
                    latency_ms=round(latency_ms, 2),
                    detections=detections,
                    image_shape=img_orig.shape
                )
            )

        if not warmup:
            self.frame_count += len(images)
            self.total_latency += latency_ms
            self.last_latency = latency_ms
            self.max_latency = max(self.max_latency, latency_ms)

        return outputs

    def get_stats(self):
        avg_latency = (
            self.total_latency / self.frame_count
            if self.frame_count > 0 else 0.0
        )

        return {
            "device": self.device,
            "fp16": self.fp16,
            "frames_processed": self.frame_count,
            "average_latency_ms": round(avg_latency, 2),
            "last_latency_ms": round(self.last_latency, 2),
            "max_latency_ms": round(self.max_latency, 2),
            "model_path": self.model_path,
            "allowed_classes": self.allowed_classes
        }

    @staticmethod
    def to_dict(output: DetectionOutput) -> Dict[str, Any]:
        return {
            "latency_ms": output.latency_ms,
            "image_shape": output.image_shape,
            "detections": [
                {
                    "bbox": det.bbox.tolist(),
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name
                }
                for det in output.detections
            ]
        }

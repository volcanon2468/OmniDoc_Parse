from ultralytics import YOLO
import numpy as np
import cv2
from typing import Dict, List, Tuple
from pathlib import Path
import torch
import sys
from app.utils.device_utils import get_device

class ObjectDetector:

    def __init__(self, config: Dict):
        self.config = config
        self.device, self.use_pin_memory = get_device()
        self.model = YOLO(self.config.get('model_path', 'yolov8n.pt'))
        self.model.to(self.device)
        self._warmup()

    def _warmup(self):
        if self.model:
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_image, verbose=False)

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        scale = 640.0 / max(height, width)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            new_width = max(32, new_width // 32 * 32)
            new_height = max(32, new_height // 32 * 32)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return (image, scale)

    def _filter_detections(self, detections: List[Dict], min_area: int=None) -> List[Dict]:
        min_area = min_area if min_area is not None else self.config.get('min_area', 100)
        filtered = []
        for det in detections:
            area = (det['box'][2] - det['box'][0]) * (det['box'][3] - det['box'][1])
            if area >= min_area:
                filtered.append(det)
        return filtered

    def _annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            label = det['class']
            conf = det['confidence']

            h = hash(label)
            color = (h % 256, (h // 256) % 256, (h // 65536) % 256)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f'{label} {conf:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated

    def detect(self, image_data, annotate: bool=True, confidence_threshold: float=None, iou_threshold: float=None) -> Dict:
        conf = confidence_threshold if confidence_threshold is not None else self.config.get('confidence_threshold', 0.25)
        iou = iou_threshold if iou_threshold is not None else self.config.get('iou_threshold', 0.45)
        try:
            if isinstance(image_data, (str, Path)):
                image = cv2.imread(str(image_data))
            elif isinstance(image_data, np.ndarray):
                image = image_data
            elif isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid or corrupted image format')
            processed_image, scale = self._preprocess_image(image)
            results = self.model.predict(processed_image, conf=conf, iou=iou, device=self.device, verbose=False)
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if scale < 1:
                        x1, x2 = (x1 / scale, x2 / scale)
                        y1, y2 = (y1 / scale, y2 / scale)
                    detection = {'box': [x1, y1, x2, y2], 'confidence': float(box.conf[0]), 'class': result.names[int(box.cls[0])]}
                    detections.append(detection)
            detections = self._filter_detections(detections)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            max_dets = self.config.get('max_detections', 100)
            if max_dets > 0:
                detections = detections[:max_dets]
            class_counts = {}
            for det in detections:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            result = {'detections': detections, 'class_counts': class_counts, 'total_objects': len(detections), 'device_used': self.device}
            if annotate:
                result['annotated_image'] = self._annotate_image(image, detections)
            return result
        except Exception as e:
            raise Exception(f'Error in object detection: {str(e)}')
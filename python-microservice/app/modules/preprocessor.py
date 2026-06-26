import cv2
import numpy as np
from typing import Dict, Tuple
from PIL import Image
from pathlib import Path
import io

class ImagePreprocessor:

    def __init__(self, config: Dict):
        self.config = config
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _load_image(self, image_data) -> np.ndarray:
        if isinstance(image_data, (str, Path)):
            image = cv2.imread(str(image_data))
        elif isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, np.ndarray):
            image = image_data.copy()
        else:
            bytes_data = image_data.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid or corrupted image format')
        return image

    def _resize_image(self, image: np.ndarray, max_dimension: int) -> np.ndarray:
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def _detect_and_correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return (image, 0.0)
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle = 0.0
        if abs(median_angle) > 0.5:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return (rotated, median_angle)
        return (image, 0.0)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _denoise_image(self, image: np.ndarray, noise_threshold: int) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
        if noise_level > noise_threshold:
            return cv2.bilateralFilter(image, 9, 75, 75)
        return image

    def _binarize(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_score = self._calculate_quality_score(otsu)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        adaptive_score = self._calculate_quality_score(adaptive)
        if otsu_score > adaptive_score:
            return (otsu, 'otsu')
        return (adaptive, 'adaptive')

    def _calculate_quality_score(self, binary_image: np.ndarray) -> float:
        laplacian_var = cv2.Laplacian(binary_image, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        contrast = binary_image.std() / 128.0
        return 0.6 * sharpness_score + 0.4 * contrast

    def process(self, image_data, config_override: dict=None) -> Dict:
        config = config_override if config_override is not None else self.config
        max_dimension = config.get('max_dimension', 2048)
        noise_threshold = config.get('noise_threshold', 100)
        try:
            image = self._load_image(image_data)
            if image is None:
                raise ValueError('Could not load image')
            original = image.copy()
            image = self._resize_image(image, max_dimension)
            image = self._denoise_image(image, noise_threshold)
            image = self._enhance_contrast(image)
            image, skew_angle = self._detect_and_correct_skew(image)
            binary, method = self._binarize(image)
            quality_score = self._calculate_quality_score(binary)
            return {'original': original, 'enhanced': image, 'binary': binary, 'skew_angle': skew_angle, 'quality_score': quality_score, 'binarization_method': method}
        except Exception as e:
            raise Exception(f'Error in preprocessing: {str(e)}')

    def to_pil_image(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
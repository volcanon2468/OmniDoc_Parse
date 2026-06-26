import pytesseract
import easyocr
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2

class OCREngine:

    def __init__(self, config: Dict):
        self.config = config
        self.reader = None
        if self.config.get('tesseract_path'):
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_path']
        self.tesseract_available = self._test_tesseract()

    def _test_tesseract(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _ensure_easyocr(self):
        if self.reader is None:
            self.reader = easyocr.Reader(['en'])

    def _process_tesseract(self, image: np.ndarray, mode: str) -> Dict:
        if not self.tesseract_available:
            return {'success': False, 'error': 'Tesseract not available'}
        try:
            config = ''
            if mode == 'fast':
                config = '--psm 6'
            elif mode == 'balanced':
                config = '--psm 6 --oem 1'
            else:
                config = '--psm 3 --oem 1'
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            words = []
            confidences = []
            boxes = []
            for i in range(len(data['text'])):
                if float(data['conf'][i]) > 0:
                    word = data['text'][i].strip()
                    if word:
                        words.append(word)
                        confidences.append(float(data['conf'][i]))
                        boxes.append((data['left'][i], data['top'][i], data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]))
            if words:
                weights = [len(word) for word in words]
                total_weight = sum(weights)
                weighted_confidence = sum((c * w for c, w in zip(confidences, weights))) / total_weight
            else:
                weighted_confidence = 0
            return {'text': ' '.join(words), 'confidence': weighted_confidence, 'words': words, 'word_confidences': confidences, 'boxes': boxes, 'engine': 'tesseract'}
        except Exception as e:
            raise Exception(f'Tesseract error: {str(e)}')

    def _process_easyocr(self, image: np.ndarray) -> Dict:
        try:
            self._ensure_easyocr()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.reader.readtext(image)
            words = []
            confidences = []
            boxes = []
            min_conf = self.config.get('easyocr_min_confidence', 10.0) / 100.0
            for bbox, text, conf in results:
                if conf > min_conf:
                    words.append(text)
                    confidences.append(conf * 100.0)
                    x1, y1 = map(int, bbox[0])
                    x3, y3 = map(int, bbox[2])
                    boxes.append((x1, y1, x3, y3))
            confidence = np.mean(confidences) if confidences else 0
            return {'text': ' '.join(words), 'confidence': confidence, 'words': words, 'word_confidences': confidences, 'boxes': boxes, 'engine': 'easyocr'}
        except Exception as e:
            raise Exception(f'EasyOCR error: {str(e)}')

    def _clean_text(self, text: str) -> str:
        text = ' '.join(text.split())
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        text = ' '.join(text.split())
        return text

    def process(self, image_data: Dict, mode: str='balanced') -> Dict:
        try:
            if 'binary' in image_data:
                image = image_data['binary']
            else:
                image = image_data.get('enhanced', image_data)
            results = {'tesseract': None, 'easyocr': None}
            final_result = None
            if self.tesseract_available:
                try:
                    results['tesseract'] = self._process_tesseract(image, mode)
                    final_result = results['tesseract']
                except Exception:
                    pass
            should_use_easyocr = False
            if not self.tesseract_available:
                should_use_easyocr = True
            elif mode == 'precise':
                should_use_easyocr = True
            elif mode != 'fast':
                min_confidence = self.config.get('min_confidence', 60)
                if not final_result:
                    should_use_easyocr = True
                elif final_result.get('confidence', 0) < min_confidence:
                    should_use_easyocr = True
            if should_use_easyocr:
                try:
                    results['easyocr'] = self._process_easyocr(image)
                    if not final_result or results['easyocr']['confidence'] > final_result.get('confidence', 0):
                        final_result = results['easyocr']
                except Exception:
                    pass
            if final_result:
                final_result['text'] = self._clean_text(final_result['text'])
                final_result['word_count'] = len(final_result['words'])
                final_result['char_count'] = len(final_result['text'])
                return final_result
            else:
                raise Exception('No OCR engine produced results')
        except Exception as e:
            raise Exception(f'Error in OCR processing: {str(e)}')
import numpy as np
import cv2
import pytesseract
import easyocr
from unittest import mock
from src.modules.ocr_engine import OCREngine

def make_test_image():
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, '42', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

@mock.patch('pytesseract.get_tesseract_version')
@mock.patch('pytesseract.image_to_data')
@mock.patch('easyocr.Reader')
def test_ocr_engine_dual(mock_easy_reader, mock_tess_data, mock_tess_version):

    mock_tess_version.return_value = '4.1.0'

    mock_tess_data.return_value = {
        'text': ['42'],
        'conf': ['95'],
        'left': [10],
        'top': [10],
        'width': [50],
        'height': [30]
    }

    instance = mock_easy_reader.return_value
    instance.readtext.return_value = [([[0,0],[10,0],[10,10],[0,10]], '42', 0.9)]

    config = {"tesseract_path": None}
    engine = OCREngine(config)

    img = make_test_image()
    result = engine.process({'binary': img}, mode='balanced')

    assert result['success'] is True
    assert 'text' in result
    assert '42' in result['text']

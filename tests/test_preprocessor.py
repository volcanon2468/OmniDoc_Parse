import numpy as np
import cv2
from src.modules.preprocessor import ImagePreprocessor

def test_preprocessor_basic():
    config = {"max_dimension": 512, "noise_threshold": 10}
    pre = ImagePreprocessor(config)

    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, 'TEST', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

    results = pre.process(img)

    assert 'binary' in results
    assert 'enhanced' in results
    assert isinstance(results['quality_score'], float)
    assert results['quality_score'] >= 0

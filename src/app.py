import streamlit as st
import warnings
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'python-microservice'))
from app.modules.preprocessor import ImagePreprocessor
from app.modules.ocr_engine import OCREngine
from app.modules.nlp_processor import NLPProcessor
from app.modules.object_detector import ObjectDetector
from utils.session_state import init_session_state, update_history
from app.core.config import settings
from utils.display import ResultsDisplay
st.set_page_config(page_title='Advanced Image Recognition & Information Extraction', page_icon='🔍', layout='wide', initial_sidebar_state='expanded')

@st.cache_resource
def initialize_components():
    config = {'preprocessing': {'max_dimension': settings.ai_max_dimension, 'noise_threshold': settings.ai_noise_threshold}, 'ocr': {'tesseract_path': settings.ai_tesseract_path, 'min_confidence': settings.ai_min_confidence}, 'nlp': {'top_keywords': settings.ai_top_keywords, 'min_keyword_length': settings.ai_min_keyword_length}, 'object_detection': {'model_path': settings.yolo_model_path, 'confidence_threshold': settings.ai_yolo_confidence, 'iou_threshold': settings.ai_yolo_iou, 'min_area': settings.ai_yolo_min_area}}
    preprocessor = ImagePreprocessor(config['preprocessing'])
    ocr_engine = OCREngine(config['ocr'])
    nlp_processor = NLPProcessor(config['nlp'])
    object_detector = ObjectDetector(config['object_detection'])
    return (preprocessor, ocr_engine, nlp_processor, object_detector)

def main():
    warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
    init_session_state()
    preprocessor, ocr_engine, nlp_processor, object_detector = initialize_components()
    st.sidebar.title('Processing Options')
    enable_preprocessing = st.sidebar.checkbox('Enable Preprocessing', value=True)
    enable_ocr = st.sidebar.checkbox('Enable OCR', value=True)
    enable_nlp = st.sidebar.checkbox('Enable NLP Analysis', value=True)
    enable_object_detection = st.sidebar.checkbox('Enable Object Detection', value=True)
    mode = st.sidebar.radio('Processing Mode', ['Fast', 'Balanced', 'Precise'], help='Fast: Quick results, lower accuracy\nBalanced: Good balance\nPrecise: Best accuracy, slower')
    st.title('Advanced Image Recognition & Information Extraction')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        if st.button('Process Image'):
            with st.spinner('Processing image...'):
                try:
                    results = {}
                    stage_times = {}
                    display = ResultsDisplay()
                    t0 = time.time()
                    preprocessed = preprocessor.process(uploaded_file)
                    stage_times['preprocessing'] = round(time.time() - t0, 3)
                    if enable_preprocessing:
                        results['preprocessed'] = preprocessed
                    if enable_ocr:
                        t0 = time.time()
                        results['ocr'] = ocr_engine.process(preprocessed, mode=mode.lower())
                        stage_times['ocr'] = round(time.time() - t0, 3)
                        if enable_nlp and results['ocr'].get('text'):
                            t0 = time.time()
                            results['nlp'] = nlp_processor.process(results['ocr']['text'])
                            stage_times['nlp'] = round(time.time() - t0, 3)
                    if enable_object_detection:
                        t0 = time.time()
                        results['objects'] = object_detector.detect(uploaded_file, annotate=True)
                        stage_times['object_detection'] = round(time.time() - t0, 3)
                    results['stage_times'] = stage_times
                    results['total_time'] = round(sum(stage_times.values()), 3)
                    results['success'] = True
                    update_history(results)
                    display.show_results(results)
                except Exception as e:
                    st.error(f'An error occurred: {str(e)}')
if __name__ == '__main__':
    main()
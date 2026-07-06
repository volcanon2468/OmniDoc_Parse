import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class ResultsDisplay:

    def __init__(self):
        self.tabs = None

    def _display_performance(self, results: Dict):
        st.subheader('Processing Performance')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Time', f"{results.get('total_time', 0):.2f}s")
        with col2:
            if 'ocr' in results:
                ocr = results['ocr']
                engine = ocr.get('engine', 'unknown')
                confidence = ocr.get('confidence', 0)
                st.metric(f'{engine.title()} Confidence', f'{confidence:.1f}%')
        with col3:
            if 'objects' in results:
                st.metric('Objects Detected', str(results['objects'].get('total_objects', 0)))
        if 'stage_times' in results:
            df = pd.DataFrame({'Stage': list(results['stage_times'].keys()), 'Time (s)': list(results['stage_times'].values())})
            fig = px.bar(df, x='Stage', y='Time (s)', title='Processing Time Breakdown')
            st.plotly_chart(fig)

    def _display_images(self, results: Dict):
        st.subheader('Image Analysis')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Original Image')
            if 'original' in results:
                st.image(results['original'])
        with col2:
            st.write('Enhanced Image')
            if 'enhanced' in results:
                st.image(results['enhanced'])
        if 'objects' in results and 'annotated_image' in results['objects']:
            st.write('Object Detection Results')
            st.image(results['objects']['annotated_image'])

    def _display_ocr_results(self, results: Dict):
        if 'ocr' not in results:
            st.warning('No OCR results available')
            return
        ocr = results['ocr']
        st.subheader('Text Extraction Results')
        st.write('Extracted Text:')
        st.text_area('OCR Text', value=ocr.get('text', ''), height=200, label_visibility='hidden')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Word Count', str(ocr.get('word_count', 0)))
        with col2:
            st.metric('Character Count', str(ocr.get('char_count', 0)))
        with col3:
            engine = ocr.get('engine', 'unknown')
            st.metric('Engine Used', engine.title())
        with col4:
            confidence = ocr.get('confidence', 0)
            st.metric(f'{engine.title()} Confidence', f'{confidence:.1f}%')
        if 'words' in ocr and 'word_confidences' in ocr:
            df = pd.DataFrame({'Word': ocr['words'], 'Confidence': ocr['word_confidences']})
            fig = px.scatter(df, x='Word', y='Confidence', title='Word Recognition Confidence')
            st.plotly_chart(fig)

    def _display_nlp_analysis(self, results: Dict):
        if 'nlp' not in results:
            st.warning('No NLP analysis available')
            return
        nlp = results['nlp']
        if not nlp.get('success', True):
            st.error(f"NLP processing failed: {nlp.get('error', 'Unknown error')}")
            return
        st.subheader('Text Analysis')
        if 'classification' in nlp:
            st.write('Document Type:')
            class_result = nlp['classification']
            st.info(f"Detected type: {class_result['type'].title()} (Confidence: {class_result['confidence']:.1%})")
            if 'scores' in class_result:
                df = pd.DataFrame({'Type': list(class_result['scores'].keys()), 'Score': list(class_result['scores'].values())})
                fig = px.bar(df, x='Type', y='Score', title='Document Type Scores')
                st.plotly_chart(fig)
        if 'entities' in nlp:
            st.write('Named Entities:')
            if nlp['entities']:
                for entity_type, entities in nlp['entities'].items():
                    if entities:
                        with st.expander(f'{entity_type} ({len(entities)})'):
                            for ent in entities:
                                st.write(f"• {ent['text']}")
            else:
                st.info('No named entities detected.')
        if 'patterns' in nlp:
            st.write('Extracted Patterns:')
            patterns = nlp['patterns']
            has_patterns = any(patterns.values())
            if has_patterns:
                for pattern_type, matches in patterns.items():
                    if matches:
                        with st.expander(f'{pattern_type.title()} ({len(matches)})'):
                            for match in matches:
                                st.write(f'• {match}')
            else:
                st.info('No patterns (emails, phones, URLs, dates) found in the text.')
        if 'keywords' in nlp and nlp['keywords']:
            st.write('Key Terms:')
            df = pd.DataFrame(nlp['keywords'])
            fig = px.bar(df, x='word', y='score', title='Keyword Importance')
            st.plotly_chart(fig)
        elif 'keywords' in nlp and (not nlp['keywords']):
            st.info('No keywords extracted from the text.')
        if 'statistics' in nlp:
            stats = nlp['statistics']
            cols = st.columns(3)
            with cols[0]:
                st.metric('Sentences', str(stats['sentence_count']))
            with cols[1]:
                st.metric('Avg. Words/Sentence', f"{stats['avg_sentence_length']:.1f}")
            with cols[2]:
                st.metric('Unique Words', str(stats['unique_words']))

    def _display_object_detection(self, results: Dict):
        if 'objects' not in results:
            st.warning('No object detection results available')
            return
        objects = results['objects']
        st.subheader('Object Detection')
        if 'detections' in objects:
            detections = objects['detections']
            if detections:
                data = []
                for det in detections:
                    data.append({'Object': det['class'], 'Confidence': f"{det['confidence']:.1%}", 'Box': f"({int(det['box'][0])}, {int(det['box'][1])}, {int(det['box'][2])}, {int(det['box'][3])})"})
                df = pd.DataFrame(data)
                st.dataframe(df)
                if 'class_counts' in objects:
                    df_counts = pd.DataFrame({'Class': list(objects['class_counts'].keys()), 'Count': list(objects['class_counts'].values())})
                    fig = px.pie(df_counts, values='Count', names='Class', title='Detected Object Classes')
                    st.plotly_chart(fig)

    def _prepare_download(self, results: Dict):
        download_data = results.copy()
        image_keys = ['original', 'enhanced', 'binary', 'annotated_image']
        for key in image_keys:
            if key in download_data:
                del download_data[key]
            if 'objects' in download_data and key in download_data['objects']:
                del download_data['objects'][key]
        download_data = self._convert_numpy_types(download_data)
        json_str = json.dumps(download_data, indent=2, cls=NumpyEncoder)
        st.download_button('Download Results (JSON)', json_str, 'results.json', 'application/json')

    def _convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def show_results(self, results: Dict):
        self.tabs = st.tabs(['Performance', 'Images', 'OCR Results', 'Text Analysis', 'Objects', 'Download'])
        with self.tabs[0]:
            self._display_performance(results)
        with self.tabs[1]:
            self._display_images(results)
        with self.tabs[2]:
            self._display_ocr_results(results)
        with self.tabs[3]:
            self._display_nlp_analysis(results)
        with self.tabs[4]:
            self._display_object_detection(results)
        with self.tabs[5]:
            self._prepare_download(results)
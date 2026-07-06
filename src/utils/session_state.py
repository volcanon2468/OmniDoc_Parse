import streamlit as st
from typing import Dict
import time

def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = []
    if 'success_count' not in st.session_state:
        st.session_state.success_count = 0
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0

def update_history(results: Dict):
    results['timestamp'] = time.time()
    st.session_state.history.append(results)
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)
    if 'processing_time' in results:
        st.session_state.processing_times.append(results['processing_time'])
        if len(st.session_state.processing_times) > 100:
            st.session_state.processing_times.pop(0)
    if results.get('success', False):
        st.session_state.success_count += 1
    else:
        st.session_state.error_count += 1

def get_performance_stats() -> Dict:
    if not st.session_state.processing_times:
        return {'avg_time': 0, 'success_rate': 0, 'total_processed': 0}
    return {'avg_time': sum(st.session_state.processing_times) / len(st.session_state.processing_times), 'success_rate': st.session_state.success_count / (st.session_state.success_count + st.session_state.error_count), 'total_processed': st.session_state.success_count + st.session_state.error_count}
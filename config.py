"""
Configuration file for Faculty Workload & Timetable Assistant
"""

# Model configurations
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "microsoft/DialoGPT-medium",
    "max_length": 512,
    "temperature": 0.7
}

# Vector store configurations
VECTOR_STORE_CONFIG = {
    "persist_directory": "./chroma_db",
    "search_kwargs": {"k": 3}
}

# Data file paths
DATA_PATHS = {
    "faculty_data": "data/faculty_data.csv",
    "timetable_data": "data/timetable_data.csv"
}

# Streamlit configurations
STREAMLIT_CONFIG = {
    "page_title": "Faculty Workload & Timetable Assistant",
    "page_icon": "🎓",
    "layout": "wide"
}


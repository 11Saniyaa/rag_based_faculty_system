import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import traceback
import re
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
import io
import hashlib
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Faculty Workload & Timetable Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FacultyWorkloadAssistant:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.faculty_data = None
        self.timetable_data = None
        self.cache = {}
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.performance_metrics = {
            'total_questions': 0,
            'cache_hits': 0,
            'average_response_time': 0,
            'error_count': 0,
            'last_reset': time.time()
        }
        logger.info(f"Initialized FacultyWorkloadAssistant with session ID: {self.session_id}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            health = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'data_loaded': self.faculty_data is not None and self.timetable_data is not None,
                'vectorstore_ready': self.vectorstore is not None,
                'cache_size': len(self.cache),
                'performance_metrics': self.performance_metrics.copy()
            }
            
            # Check for issues
            issues = []
            if not health['data_loaded']:
                issues.append("Data not loaded")
            if not health['vectorstore_ready']:
                issues.append("Vector store not ready")
            if health['performance_metrics']['error_count'] > 10:
                issues.append("High error count")
            
            if issues:
                health['status'] = 'degraded'
                health['issues'] = issues
            
            return health
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_questions': 0,
            'cache_hits': 0,
            'average_response_time': 0,
            'error_count': 0,
            'last_reset': time.time()
        }
        logger.info("Performance metrics reset")
        
    def _validate_input(self, question: str) -> Tuple[bool, str]:
        """Validate user input for security and format"""
        if not question or not isinstance(question, str):
            return False, "Please provide a valid question."
        
        if len(question.strip()) < 3:
            return False, "Question too short. Please provide a more detailed question."
        
        if len(question) > 1000:
            return False, "Question too long. Please keep it under 1000 characters."
        
        # Basic security checks
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                logger.warning(f"Potentially dangerous input detected: {question[:50]}...")
                return False, "Invalid input detected. Please rephrase your question."
        
        return True, ""
    
    @lru_cache(maxsize=100)
    def _cached_answer(self, question_hash: str, question: str) -> str:
        """Cache answers for performance"""
        return self._process_question(question)
    
    def _process_question(self, question: str) -> str:
        """Process question without caching"""
        try:
            # Extract question information
            question_info = self._extract_question_info(question)
            
            # Get context from vector store
            context = self._get_context(question)
            
            # Analyze and answer the question
            return self._analyze_and_answer_question(question, context, question_info)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            logger.error(traceback.format_exc())
            return f"I encountered an error while processing your question. Please try again or contact support if the issue persists."
        
    def load_data(self, faculty_file=None, timetable_file=None):
        """Load faculty and timetable data from CSV files or uploaded files"""
        try:
            # Load faculty data
            if faculty_file is not None:
                self.faculty_data = faculty_file
            elif os.path.exists("data/faculty_data.csv"):
                self.faculty_data = pd.read_csv("data/faculty_data.csv")
            else:
                # Create sample faculty data
                self.faculty_data = self.create_sample_faculty_data()
                
            # Load timetable data
            if timetable_file is not None:
                self.timetable_data = timetable_file
            elif os.path.exists("data/timetable_data.csv"):
                self.timetable_data = pd.read_csv("data/timetable_data.csv")
            else:
                # Create sample timetable data
                self.timetable_data = self.create_sample_timetable_data()
                
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def validate_csv_data(self, df, data_type):
        """Enhanced CSV data validation with comprehensive checks"""
        try:
            if data_type == "faculty":
                required_columns = ['faculty_id', 'name', 'department', 'designation', 'max_hours_per_week', 'specialization', 'availability']
            else:  # timetable
                required_columns = ['course_id', 'course_name', 'faculty_id', 'day', 'start_time', 'end_time', 'room', 'credits']
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            # Check if data is empty
            if df.empty:
                return False, "CSV file is empty"
            
            # Check for duplicate IDs
            if data_type == "faculty":
                if df['faculty_id'].duplicated().any():
                    return False, "Duplicate faculty_id found. Each faculty must have a unique ID."
            else:
                if df['course_id'].duplicated().any():
                    return False, "Duplicate course_id found. Each course must have a unique ID."
            
            # Check for null values in critical columns
            critical_columns = ['faculty_id', 'name'] if data_type == "faculty" else ['course_id', 'course_name', 'faculty_id']
            null_columns = df[critical_columns].isnull().any()
            if null_columns.any():
                null_cols = null_columns[null_columns].index.tolist()
                return False, f"Null values found in critical columns: {', '.join(null_cols)}"
            
            # Validate data types and formats
            if data_type == "faculty":
                # Check max_hours_per_week is numeric
                if not pd.api.types.is_numeric_dtype(df['max_hours_per_week']):
                    return False, "max_hours_per_week must be numeric"
                
                # Check for negative hours
                if (df['max_hours_per_week'] < 0).any():
                    return False, "max_hours_per_week cannot be negative"
                
                # Check availability format (relaxed pattern)
                availability_pattern = r'.*[Mm]on.*[Ff]ri.*\d+.*\d+.*'
                invalid_availability = df[~df['availability'].str.match(availability_pattern, na=False)]
                if not invalid_availability.empty:
                    logger.warning(f"Some availability formats may be non-standard: {invalid_availability.index.tolist()[:3]}")
            
            else:  # timetable
                # Check credits is numeric
                if not pd.api.types.is_numeric_dtype(df['credits']):
                    return False, "credits must be numeric"
                
                # Check for negative credits
                if (df['credits'] < 0).any():
                    return False, "credits cannot be negative"
                
                # Check day format
                valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
                invalid_days = df[~df['day'].isin(valid_days)]
                if not invalid_days.empty:
                    return False, f"Invalid day format found in rows: {invalid_days.index.tolist()[:5]}"
                
                # Check time format
                time_pattern = r'^\d{1,2}:\d{2}$'
                invalid_times = df[~df['start_time'].str.match(time_pattern, na=False) | ~df['end_time'].str.match(time_pattern, na=False)]
                if not invalid_times.empty:
                    return False, f"Invalid time format found in rows: {invalid_times.index.tolist()[:5]}"
            
            logger.info(f"Data validation passed for {data_type} data with {len(df)} records")
            return True, f"Data validation passed. {len(df)} records validated successfully."
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def process_uploaded_csv(self, uploaded_file, data_type):
        """Process uploaded CSV file"""
        try:
            # Read CSV from uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Validate the data
            is_valid, message = self.validate_csv_data(df, data_type)
            if not is_valid:
                return None, message
            
            return df, "CSV file processed successfully"
            
        except Exception as e:
            return None, f"Error processing CSV file: {str(e)}"
    
    def create_sample_faculty_data(self):
        """Create sample faculty data"""
        faculty_data = {
            'faculty_id': ['F001', 'F002', 'F003', 'F004', 'F005'],
            'name': ['Dr. John Smith', 'Dr. Sarah Johnson', 'Dr. Michael Brown', 'Dr. Emily Davis', 'Dr. Robert Wilson'],
            'department': ['Computer Science', 'Mathematics', 'Physics', 'Computer Science', 'Mathematics'],
            'designation': ['Professor', 'Associate Professor', 'Assistant Professor', 'Professor', 'Associate Professor'],
            'max_hours_per_week': [40, 40, 40, 40, 40],
            'specialization': ['Machine Learning', 'Statistics', 'Quantum Physics', 'Data Science', 'Algebra'],
            'availability': ['Mon-Fri 9AM-5PM', 'Mon-Fri 10AM-6PM', 'Mon-Fri 8AM-4PM', 'Mon-Fri 9AM-5PM', 'Mon-Fri 10AM-6PM']
        }
        return pd.DataFrame(faculty_data)
    
    def create_sample_timetable_data(self):
        """Create sample timetable data"""
        timetable_data = {
            'course_id': ['CS101', 'CS102', 'MATH201', 'PHYS301', 'CS301'],
            'course_name': ['Introduction to Programming', 'Data Structures', 'Calculus II', 'Quantum Mechanics', 'Machine Learning'],
            'faculty_id': ['F001', 'F001', 'F002', 'F003', 'F004'],
            'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'start_time': ['09:00', '10:00', '11:00', '14:00', '15:00'],
            'end_time': ['10:30', '11:30', '12:30', '15:30', '16:30'],
            'room': ['Room A101', 'Room A102', 'Room B201', 'Room C301', 'Room A103'],
            'credits': [3, 3, 4, 4, 3]
        }
        return pd.DataFrame(timetable_data)
    
    def setup_vectorstore(self):
        """Setup vector store for RAG"""
        try:
            # Create documents from faculty and timetable data
            documents = []
            
            # Add faculty information
            for _, row in self.faculty_data.iterrows():
                doc = f"""
                Faculty: {row['name']}
                Department: {row['department']}
                Designation: {row['designation']}
                Specialization: {row['specialization']}
                Availability: {row['availability']}
                Max Hours per Week: {row['max_hours_per_week']}
                """
                documents.append(doc)
            
            # Add timetable information
            for _, row in self.timetable_data.iterrows():
                doc = f"""
                Course: {row['course_name']} ({row['course_id']})
                Faculty: {row['faculty_id']}
                Day: {row['day']}
                Time: {row['start_time']} - {row['end_time']}
                Room: {row['room']}
                Credits: {row['credits']}
                """
                documents.append(doc)
            
            # Create a simple mock vector store for now
            class MockVectorStore:
                def __init__(self, docs):
                    self.docs = docs
                
                def similarity_search(self, query, k=3):
                    # Simple keyword matching
                    query_lower = query.lower()
                    results = []
                    for doc in self.docs:
                        if any(word in doc.lower() for word in query_lower.split()):
                            results.append(type('obj', (object,), {'page_content': doc}))
                    return results[:k]
            
            self.vectorstore = MockVectorStore(documents)
            return True
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
            return False
    
    def setup_llm(self):
        """Setup LLM for question answering"""
        try:
            # Use a simple rule-based approach for now
            # This will be more reliable than the complex model setup
            self.llm = "rule_based"
            return True
        except Exception as e:
            st.error(f"Error setting up LLM: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Setup question answering chain"""
        try:
            # For now, we'll use a simple approach without the complex chain
            # This will be handled in the answer_question method
            return True
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            return False
    
    def answer_question(self, question: str) -> str:
        """Enhanced answer method with validation, caching, and error handling"""
        try:
            # Validate input
            is_valid, error_msg = self._validate_input(question)
            if not is_valid:
                logger.warning(f"Invalid input: {error_msg}")
                return error_msg
            
            # Check if system is initialized
            if self.faculty_data is None or self.timetable_data is None:
                logger.info("System not initialized, re-initializing...")
                self.load_data()
            
            if self.vectorstore is None:
                self.setup_vectorstore()
            
            # Create cache key
            question_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
            
            # Update performance metrics
            self.performance_metrics['total_questions'] += 1
            
            # Check cache first
            if question_hash in self.cache:
                self.performance_metrics['cache_hits'] += 1
                logger.info(f"Cache hit for question: {question[:50]}...")
                return self.cache[question_hash]
            
            # Process question
            start_time = time.time()
            
            # Get relevant documents from vector store
            docs = self.vectorstore.similarity_search(question, k=5)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Use intelligent question analysis
            answer = self._analyze_and_answer_question(question, context)
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            total_time = self.performance_metrics['average_response_time'] * (self.performance_metrics['total_questions'] - 1)
            self.performance_metrics['average_response_time'] = (total_time + processing_time) / self.performance_metrics['total_questions']
            
            # Cache the answer
            self.cache[question_hash] = answer
            
            # Log performance
            logger.info(f"Question processed in {processing_time:.2f}s: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            logger.error(f"Error in answer_question: {str(e)}")
            logger.error(traceback.format_exc())
            return "I apologize, but I encountered an unexpected error while processing your question. Please try again or contact support if the issue persists."
    
    def _analyze_and_answer_question(self, question: str, context: str) -> str:
        """Intelligently analyze and answer any type of question"""
        question_lower = question.lower()
        
        # Extract key information from the question
        question_info = self._extract_question_info(question)
        
        # Route to appropriate handler based on question analysis
        if question_info['type'] == 'availability':
            return self._get_availability_info(question, context)
        elif question_info['type'] == 'comparison':
            return self._get_comparison_info(question, context, question_info)
        elif question_info['type'] == 'course_search':
            return self._get_course_search_info(question, context)
        elif question_info['type'] == 'schedule':
            return self._get_schedule_info(question, context)
        elif question_info['type'] == 'workload':
            return self._get_workload_info(question, context)
        elif question_info['type'] == 'department_workload':
            return self._get_department_workload_info(question, context)
        elif question_info['type'] == 'conflict':
            return self._get_conflict_info(question, context)
        elif question_info['type'] == 'department':
            return self._get_department_info(question, context)
        elif question_info['type'] == 'listing':
            return self._get_listing_info(question, context, question_info)
        elif question_info['type'] == 'statistics':
            return self._get_statistics_info(question, context, question_info)
        elif question_info['type'] == 'search':
            return self._get_search_info(question, context, question_info)
        else:
            return self._get_comprehensive_answer(question, context, question_info)
    
    def _extract_question_info(self, question: str) -> dict:
        """Extract key information from the question"""
        question_lower = question.lower()
        
        # Initialize question info
        info = {
            'type': 'general',
            'faculty': [],
            'department': None,
            'day': None,
            'time': None,
            'comparison_type': None,
            'action': None,
            'keywords': []
        }
        
        # Extract faculty names
        faculty_names = self.faculty_data['name'].str.lower().tolist()
        for name in faculty_names:
            if name in question_lower:
                info['faculty'].append(name.title())
        
        # Extract departments
        departments = ['mathematics', 'computer science', 'physics', 'math', 'cs']
        for dept in departments:
            if dept in question_lower:
                if dept in ['math', 'mathematics']:
                    info['department'] = 'Mathematics'
                elif dept in ['cs', 'computer science']:
                    info['department'] = 'Computer Science'
                elif dept == 'physics':
                    info['department'] = 'Physics'
                break
        
        # Extract days
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        for day in days:
            if day in question_lower:
                info['day'] = day.capitalize()
                break
        
        # Determine question type with better priority order
        if any(word in question_lower for word in ['conflict', 'clash', 'overlap', 'problem']):
            info['type'] = 'conflict'
        elif any(word in question_lower for word in ['free', 'available', 'not teaching', 'busy']):
            info['type'] = 'availability'
        elif any(word in question_lower for word in ['highest', 'most', 'maximum', 'lowest', 'least', 'minimum', 'more', 'less', 'compare']):
            info['type'] = 'comparison'
            if any(word in question_lower for word in ['highest', 'most', 'maximum']):
                info['comparison_type'] = 'highest'
            elif any(word in question_lower for word in ['lowest', 'least', 'minimum']):
                info['comparison_type'] = 'lowest'
        elif any(word in question_lower for word in ['who teaches', 'which faculty teaches', 'teaches in room', 'teaches machine learning', 'teaches data structures']):
            info['type'] = 'course_search'
        elif any(word in question_lower for word in ['free', 'available', 'not teaching', 'busy', 'is free', 'are free']) or ('teaching' in question_lower and any(day in question_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])) or (any(day in question_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']) and any(word in question_lower for word in ['is', 'are', 'free', 'available'])):
            info['type'] = 'availability'
        elif any(word in question_lower for word in ['schedule', 'time', 'when', 'teach', 'teaching']) and any(name in question_lower for name in faculty_names):
            info['type'] = 'schedule'
        elif any(word in question_lower for word in ['workload', 'hours', 'courses', 'credits']) and any(name in question_lower for name in faculty_names):
            info['type'] = 'workload'
        elif any(word in question_lower for word in ['department', 'dept']) and any(word in question_lower for word in ['workload', 'courses', 'faculty']):
            info['type'] = 'department_workload'
        elif any(word in question_lower for word in ['department', 'dept']):
            info['type'] = 'department'
        elif any(word in question_lower for word in ['show', 'list', 'all', 'every', 'each']):
            info['type'] = 'listing'
        elif any(word in question_lower for word in ['total', 'sum', 'count', 'number', 'how many', 'statistics', 'average']):
            info['type'] = 'statistics'
        elif any(word in question_lower for word in ['find', 'search', 'who', 'which', 'what']):
            info['type'] = 'search'
        elif any(word in question_lower for word in ['schedule', 'time', 'when', 'teach', 'teaching']):
            info['type'] = 'schedule'
        elif any(word in question_lower for word in ['workload', 'hours', 'courses', 'credits']):
            info['type'] = 'workload'
        
        # Extract action words
        if any(word in question_lower for word in ['teach', 'teaching', 'instruct']):
            info['action'] = 'teaching'
        elif any(word in question_lower for word in ['free', 'available']):
            info['action'] = 'available'
        
        # Extract keywords
        keywords = ['room', 'course', 'subject', 'class', 'lecture', 'seminar', 'lab']
        for keyword in keywords:
            if keyword in question_lower:
                info['keywords'].append(keyword)
        
        return info
    
    def _get_schedule_info(self, question: str, context: str) -> str:
        """Get schedule information"""
        # Extract faculty name from question
        faculty_names = self.faculty_data['name'].tolist()
        for name in faculty_names:
            if name.lower() in question.lower():
                faculty_row = self.faculty_data[self.faculty_data['name'] == name]
                faculty_id = faculty_row['faculty_id'].iloc[0]
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                
                # Get faculty details
                department = faculty_row['department'].iloc[0]
                designation = faculty_row['designation'].iloc[0]
                specialization = faculty_row['specialization'].iloc[0]
                availability = faculty_row['availability'].iloc[0]
                
                if not courses.empty:
                    result = f"**Teaching Schedule for {name}**\n"
                    result += f"==========================================\n\n"
                    
                    # Faculty Information
                    result += f"**Faculty Information:**\n"
                    result += f"- Department: {department}\n"
                    result += f"- Designation: {designation}\n"
                    result += f"- Specialization: {specialization}\n"
                    result += f"- General Availability: {availability}\n\n"
                    
                    # Schedule Details
                    result += f"**Teaching Schedule:**\n"
                    for _, course in courses.iterrows():
                        result += f"• **{course['course_name']}** ({course['credits']} credits)\n"
                        result += f"  - Day: {course['day']}\n"
                        result += f"  - Time: {course['start_time']} - {course['end_time']}\n"
                        result += f"  - Room: {course['room']}\n\n"
                    
                    # Weekly Overview
                    result += f"**Weekly Overview:**\n"
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    for day in days:
                        day_courses = courses[courses['day'] == day]
                        if not day_courses.empty:
                            result += f"- {day}: "
                            course_list = []
                            for _, course in day_courses.iterrows():
                                course_list.append(f"{course['course_name']} ({course['start_time']})")
                            result += ", ".join(course_list) + "\n"
                        else:
                            result += f"- {day}: No classes\n"
                    
                    # Summary
                    total_hours = courses['credits'].sum()
                    result += f"\n**Summary:**\n"
                    result += f"- Total Courses: {len(courses)}\n"
                    result += f"- Total Credit Hours: {total_hours}\n"
                    result += f"- Teaching Days: {len(courses['day'].unique())}\n"
                    
                    return result
                else:
                    return f"**Schedule for {name}**\n\n{name} has no scheduled courses.\n\n**Faculty Information:**\n- Department: {department}\n- Designation: {designation}\n- Specialization: {specialization}\n- General Availability: {availability}"
        
        return "Please specify a faculty member name to get their schedule."
    
    def _get_workload_info(self, question: str, context: str) -> str:
        """Get workload information"""
        if "all" in question.lower() or "total" in question.lower():
            total_hours = self.timetable_data['credits'].sum()
            total_courses = len(self.timetable_data)
            total_faculty = len(self.faculty_data)
            avg_workload = total_hours / total_faculty if total_faculty > 0 else 0
            
            result = f"**Overall Workload Summary**\n"
            result += f"==========================================\n\n"
            result += f"**Statistics:**\n"
            result += f"- Total Faculty: {total_faculty}\n"
            result += f"- Total Courses: {total_courses}\n"
            result += f"- Total Credit Hours: {total_hours}\n"
            result += f"- Average Workload: {avg_workload:.1f} credit hours per faculty\n\n"
            
            # Department breakdown
            result += f"**Department Breakdown:**\n"
            for dept in self.faculty_data['department'].unique():
                dept_faculty = self.faculty_data[self.faculty_data['department'] == dept]
                dept_courses = self.timetable_data[self.timetable_data['faculty_id'].isin(dept_faculty['faculty_id'])]
                dept_hours = dept_courses['credits'].sum()
                result += f"- {dept}: {len(dept_faculty)} faculty, {len(dept_courses)} courses, {dept_hours} credit hours\n"
            
            return result
        
        # Individual faculty workload
        faculty_names = self.faculty_data['name'].tolist()
        for name in faculty_names:
            if name.lower() in question.lower():
                faculty_row = self.faculty_data[self.faculty_data['name'] == name]
                faculty_id = faculty_row['faculty_id'].iloc[0]
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                total_hours = courses['credits'].sum()
                num_courses = len(courses)
                
                # Get faculty details
                department = faculty_row['department'].iloc[0]
                designation = faculty_row['designation'].iloc[0]
                specialization = faculty_row['specialization'].iloc[0]
                max_hours = faculty_row['max_hours_per_week'].iloc[0]
                availability = faculty_row['availability'].iloc[0]
                
                # Calculate workload percentage
                workload_percentage = (total_hours / max_hours) * 100 if max_hours > 0 else 0
                
                result = f"**Workload Analysis for {name}**\n"
                result += f"==========================================\n\n"
                
                # Faculty Details
                result += f"**Faculty Information:**\n"
                result += f"- Department: {department}\n"
                result += f"- Designation: {designation}\n"
                result += f"- Specialization: {specialization}\n"
                result += f"- Availability: {availability}\n"
                result += f"- Max Hours per Week: {max_hours}\n\n"
                
                # Workload Summary
                result += f"**Workload Summary:**\n"
                result += f"- Total Courses: {num_courses}\n"
                result += f"- Total Credit Hours: {total_hours}\n"
                result += f"- Workload Utilization: {workload_percentage:.1f}% of maximum\n"
                
                # Workload Status
                if workload_percentage < 50:
                    result += f"- Status: [GREEN] Light workload (under 50% capacity)\n"
                elif workload_percentage < 80:
                    result += f"- Status: [YELLOW] Moderate workload (50-80% capacity)\n"
                else:
                    result += f"- Status: [RED] Heavy workload (over 80% capacity)\n"
                
                # Detailed Course Information
                if not courses.empty:
                    result += f"\n**Course Details:**\n"
                    for _, course in courses.iterrows():
                        result += f"• **{course['course_name']}** ({course['credits']} credits)\n"
                        result += f"  - Day: {course['day']}\n"
                        result += f"  - Time: {course['start_time']} - {course['end_time']}\n"
                        result += f"  - Room: {course['room']}\n\n"
                    
                    # Weekly Schedule Overview
                    result += f"**Weekly Schedule:**\n"
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    for day in days:
                        day_courses = courses[courses['day'] == day]
                        if not day_courses.empty:
                            result += f"- {day}: "
                            course_list = []
                            for _, course in day_courses.iterrows():
                                course_list.append(f"{course['course_name']} ({course['start_time']})")
                            result += ", ".join(course_list) + "\n"
                        else:
                            result += f"- {day}: No classes\n"
                
                # Recommendations
                result += f"\n**Recommendations:**\n"
                if workload_percentage < 50:
                    result += f"- Consider assigning additional courses to optimize faculty utilization\n"
                elif workload_percentage > 90:
                    result += f"- Consider reducing workload to prevent burnout\n"
                else:
                    result += f"- Current workload is well-balanced\n"
                
                return result
        
        return "Please specify a faculty member name to get their workload information."
    
    def _get_conflict_info(self, question: str, context: str) -> str:
        """Get conflict information"""
        faculty_conflicts = []
        room_conflicts = []
        
        # Check for faculty time conflicts (same faculty, overlapping times)
        for day in self.timetable_data['day'].unique():
            day_schedule = self.timetable_data[self.timetable_data['day'] == day]
            
            for i, row1 in day_schedule.iterrows():
                for j, row2 in day_schedule.iterrows():
                    if i != j and row1['faculty_id'] == row2['faculty_id']:
                        # Check for time overlap
                        start1 = datetime.strptime(row1['start_time'], '%H:%M')
                        end1 = datetime.strptime(row1['end_time'], '%H:%M')
                        start2 = datetime.strptime(row2['start_time'], '%H:%M')
                        end2 = datetime.strptime(row2['end_time'], '%H:%M')
                        
                        if (start1 < end2 and start2 < end1):
                            faculty_name = self.faculty_data[self.faculty_data['faculty_id'] == row1['faculty_id']]['name'].iloc[0]
                            faculty_conflicts.append(
                                f"**Faculty Time Conflict:** {faculty_name} on {day}\n"
                                f"- {row1['course_name']} ({row1['start_time']}-{row1['end_time']}) overlaps with\n"
                                f"- {row2['course_name']} ({row2['start_time']}-{row2['end_time']})\n"
                            )
        
        # Check for room conflicts (same room, overlapping times)
        for day in self.timetable_data['day'].unique():
            day_schedule = self.timetable_data[self.timetable_data['day'] == day]
            
            for i, row1 in day_schedule.iterrows():
                for j, row2 in day_schedule.iterrows():
                    if i != j and row1['room'] == row2['room']:
                        # Check for time overlap
                        start1 = datetime.strptime(row1['start_time'], '%H:%M')
                        end1 = datetime.strptime(row1['end_time'], '%H:%M')
                        start2 = datetime.strptime(row2['start_time'], '%H:%M')
                        end2 = datetime.strptime(row2['end_time'], '%H:%M')
                        
                        if (start1 < end2 and start2 < end1):
                            faculty1_name = self.faculty_data[self.faculty_data['faculty_id'] == row1['faculty_id']]['name'].iloc[0]
                            faculty2_name = self.faculty_data[self.faculty_data['faculty_id'] == row2['faculty_id']]['name'].iloc[0]
                            room_conflicts.append(
                                f"**Room Conflict:** {row1['room']} on {day}\n"
                                f"- {faculty1_name}: {row1['course_name']} ({row1['start_time']}-{row1['end_time']})\n"
                                f"- {faculty2_name}: {row2['course_name']} ({row2['start_time']}-{row2['end_time']})\n"
                            )
        
        # Compile results
        result = "**Conflict Analysis Report**\n"
        result += "==========================================\n\n"
        
        if faculty_conflicts or room_conflicts:
            if faculty_conflicts:
                result += "**Faculty Time Conflicts:**\n"
                result += "\n".join(faculty_conflicts) + "\n\n"
            
            if room_conflicts:
                result += "**Room Conflicts:**\n"
                result += "\n".join(room_conflicts) + "\n\n"
            
            result += f"**Summary:** {len(faculty_conflicts)} faculty conflicts, {len(room_conflicts)} room conflicts detected."
        else:
            result += "**No conflicts detected in the current timetable.**\n\n"
            result += "**System Status:** All schedules are properly organized with no overlapping times or room conflicts."
        
        return result
    
    def _get_department_info(self, question: str, context: str) -> str:
        """Get department information"""
        dept_counts = self.faculty_data['department'].value_counts()
        result = "Faculty by Department:\n"
        for dept, count in dept_counts.items():
            result += f"- {dept}: {count} faculty\n"
        return result
    
    def _get_highest_workload_info(self, question: str, context: str) -> str:
        """Get highest workload information"""
        question_lower = question.lower()
        
        # Check if asking about specific department
        if "mathematics" in question_lower or "math" in question_lower:
            dept = "Mathematics"
        elif "computer science" in question_lower or "cs" in question_lower:
            dept = "Computer Science"
        elif "physics" in question_lower:
            dept = "Physics"
        else:
            dept = None
        
        # Get faculty workload data
        faculty_workload = []
        for _, faculty in self.faculty_data.iterrows():
            if dept is None or faculty['department'] == dept:
                faculty_id = faculty['faculty_id']
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                total_hours = courses['credits'].sum()
                faculty_workload.append({
                    'name': faculty['name'],
                    'department': faculty['department'],
                    'hours': total_hours,
                    'courses': len(courses)
                })
        
        if not faculty_workload:
            return f"No faculty found in {dept} department." if dept else "No faculty data available."
        
        # Find highest workload
        highest = max(faculty_workload, key=lambda x: x['hours'])
        
        if dept:
            return f"Highest workload in {dept} department:\n- Faculty: {highest['name']}\n- Total credit hours: {highest['hours']}\n- Number of courses: {highest['courses']}"
        else:
            return f"Highest workload across all faculty:\n- Faculty: {highest['name']} ({highest['department']})\n- Total credit hours: {highest['hours']}\n- Number of courses: {highest['courses']}"
    
    def _get_lowest_workload_info(self, question: str, context: str) -> str:
        """Get lowest workload information"""
        question_lower = question.lower()
        
        # Check if asking about specific department
        if "mathematics" in question_lower or "math" in question_lower:
            dept = "Mathematics"
        elif "computer science" in question_lower or "cs" in question_lower:
            dept = "Computer Science"
        elif "physics" in question_lower:
            dept = "Physics"
        else:
            dept = None
        
        # Get faculty workload data
        faculty_workload = []
        for _, faculty in self.faculty_data.iterrows():
            if dept is None or faculty['department'] == dept:
                faculty_id = faculty['faculty_id']
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                total_hours = courses['credits'].sum()
                faculty_workload.append({
                    'name': faculty['name'],
                    'department': faculty['department'],
                    'hours': total_hours,
                    'courses': len(courses)
                })
        
        if not faculty_workload:
            return f"No faculty found in {dept} department." if dept else "No faculty data available."
        
        # Find lowest workload
        lowest = min(faculty_workload, key=lambda x: x['hours'])
        
        if dept:
            return f"Lowest workload in {dept} department:\n- Faculty: {lowest['name']}\n- Total credit hours: {lowest['hours']}\n- Number of courses: {lowest['courses']}"
        else:
            return f"Lowest workload across all faculty:\n- Faculty: {lowest['name']} ({lowest['department']})\n- Total credit hours: {lowest['hours']}\n- Number of courses: {lowest['courses']}"
    
    def _get_show_all_info(self, question: str, context: str) -> str:
        """Get show all information"""
        question_lower = question.lower()
        
        if "mathematics" in question_lower or "math" in question_lower:
            dept = "Mathematics"
        elif "computer science" in question_lower or "cs" in question_lower:
            dept = "Computer Science"
        elif "physics" in question_lower:
            dept = "Physics"
        else:
            dept = None
        
        if dept:
            # Show all faculty in specific department
            dept_faculty = self.faculty_data[self.faculty_data['department'] == dept]
            result = f"All faculty in {dept} department:\n"
            for _, faculty in dept_faculty.iterrows():
                faculty_id = faculty['faculty_id']
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                total_hours = courses['credits'].sum()
                result += f"- {faculty['name']}: {total_hours} credit hours ({len(courses)} courses)\n"
            return result
        else:
            # Show all faculty workload
            result = "All faculty workload:\n"
            for _, faculty in self.faculty_data.iterrows():
                faculty_id = faculty['faculty_id']
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                total_hours = courses['credits'].sum()
                result += f"- {faculty['name']} ({faculty['department']}): {total_hours} credit hours ({len(courses)} courses)\n"
            return result
    
    def _get_availability_info(self, question: str, context: str) -> str:
        """Get availability information"""
        question_lower = question.lower()
        
        # Extract day from question
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        target_day = None
        for day in days:
            if day in question_lower:
                target_day = day.capitalize()
                break
        
        if not target_day:
            return "Please specify a day (Monday, Tuesday, Wednesday, Thursday, or Friday) to check faculty availability."
        
        # Check if asking about specific faculty
        faculty_names = []
        if self.faculty_data is not None:
            faculty_names = [name.lower() for name in self.faculty_data['name'].tolist()]
        
        target_faculty = None
        for name in faculty_names:
            if name in question_lower:
                target_faculty = name
                break
        
        # Extract specific time from question
        target_time = None
        time_patterns = [
            r'at (\d{1,2})',  # "at 10", "at 11"
            r'(\d{1,2}):(\d{2})',  # "10:30", "11:00"
            r'(\d{1,2}) am',  # "10 am", "11 am"
            r'(\d{1,2}) pm',  # "10 pm", "11 pm"
        ]
        
        import re
        for pattern in time_patterns:
            match = re.search(pattern, question_lower)
            if match:
                if ':' in pattern:
                    target_time = match.group(1) + ':' + match.group(2)
                else:
                    hour = int(match.group(1))
                    if 'pm' in pattern and hour != 12:
                        hour += 12
                    elif 'am' in pattern and hour == 12:
                        hour = 0
                    target_time = f"{hour:02d}:00"
                break
        
        # Find faculty who are free on the specified day and time
        free_faculty = []
        teaching_faculty = []
        busy_at_time = []
        not_available = []
        
        for _, faculty in self.faculty_data.iterrows():
            faculty_id = faculty['faculty_id']
            faculty_name = faculty['name']
            general_availability = faculty['availability']
            
            # Check if faculty is generally available on the target day and time
            is_generally_available = self._check_general_availability(general_availability, target_day, target_time)
            
            if not is_generally_available:
                not_available.append(faculty_name)
                continue
            
            # Check if faculty has any courses on the target day
            day_courses = self.timetable_data[
                (self.timetable_data['faculty_id'] == faculty_id) & 
                (self.timetable_data['day'] == target_day)
            ]
            
            if day_courses.empty:
                free_faculty.append(faculty_name)
            else:
                faculty_info = {
                    'name': faculty_name,
                    'courses': day_courses['course_name'].tolist(),
                    'times': [f"{row['start_time']}-{row['end_time']}" for _, row in day_courses.iterrows()],
                    'start_times': [row['start_time'] for _, row in day_courses.iterrows()],
                    'end_times': [row['end_time'] for _, row in day_courses.iterrows()]
                }
                teaching_faculty.append(faculty_info)
                
                # Check if faculty is busy at specific time
                if target_time:
                    is_busy_at_time = False
                    for i, (start, end) in enumerate(zip(faculty_info['start_times'], faculty_info['end_times'])):
                        if self._is_time_in_range(target_time, start, end):
                            is_busy_at_time = True
                            break
                    
                    if is_busy_at_time:
                        busy_at_time.append(faculty_name)
        
        # Generate result based on whether specific time was requested
        if target_time:
            if target_faculty:
                # Specific faculty at specific time
                faculty_name = target_faculty.title()
                if faculty_name in [name for name in free_faculty if name not in busy_at_time]:
                    result = f"**Yes, {faculty_name} is FREE on {target_day} at {target_time}**\n"
                    result += f"==========================================\n\n"
                    result += f"**Faculty Information:**\n"
                    faculty_row = self.faculty_data[self.faculty_data['name'].str.lower() == target_faculty]
                    if not faculty_row.empty:
                        result += f"- Name: {faculty_row['name'].iloc[0]}\n"
                        result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                        result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                        result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                    result += f"\n**Status:** Available for meetings, consultations, or other activities at {target_time} on {target_day}."
                elif faculty_name in busy_at_time:
                    result = f"**No, {faculty_name} is NOT FREE on {target_day} at {target_time}**\n"
                    result += f"==========================================\n\n"
                    result += f"**Faculty Information:**\n"
                    faculty_row = self.faculty_data[self.faculty_data['name'].str.lower() == target_faculty]
                    if not faculty_row.empty:
                        result += f"- Name: {faculty_row['name'].iloc[0]}\n"
                        result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                        result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                        result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                    result += f"\n**Status:** Currently teaching at {target_time} on {target_day}."
                    # Show what they're teaching
                    for faculty_info in teaching_faculty:
                        if faculty_info['name'].lower() == target_faculty:
                            for i, (start, end) in enumerate(zip(faculty_info['start_times'], faculty_info['end_times'])):
                                if self._is_time_in_range(target_time, start, end):
                                    result += f"\n**Teaching:** {faculty_info['courses'][i]} ({start}-{end})"
                            break
                elif faculty_name in not_available:
                    result = f"**No, {faculty_name} is NOT AVAILABLE on {target_day} at {target_time}**\n"
                    result += f"==========================================\n\n"
                    result += f"**Faculty Information:**\n"
                    faculty_row = self.faculty_data[self.faculty_data['name'].str.lower() == target_faculty]
                    if not faculty_row.empty:
                        result += f"- Name: {faculty_row['name'].iloc[0]}\n"
                        result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                        result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                        result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                    result += f"\n**Status:** Outside working hours at {target_time} on {target_day}."
                else:
                    result = f"**Faculty {faculty_name} not found in the system.**"
            else:
                # General availability at specific time
                result = f"**Faculty Availability on {target_day} at {target_time}**\n"
                result += f"==========================================\n\n"
                
                # Find faculty free at specific time
                free_at_time = [name for name in free_faculty if name not in busy_at_time]
                teaching_at_time = [name for name in busy_at_time]
                
                result += f"**Summary for {target_time}:**\n"
                result += f"- Total Faculty: {len(self.faculty_data)}\n"
                result += f"- Free at {target_time}: {len(free_at_time)}\n"
                result += f"- Teaching at {target_time}: {len(teaching_at_time)}\n"
                result += f"- Not Available (outside working hours): {len(not_available)}\n\n"
            
                if not target_faculty:  # Only show general list if not asking about specific faculty
                    if free_at_time:
                        result += f"[FREE] **Faculty who are FREE at {target_time} on {target_day}:**\n"
                        for name in free_at_time:
                            faculty_row = self.faculty_data[self.faculty_data['name'] == name]
                            department = faculty_row['department'].iloc[0]
                            designation = faculty_row['designation'].iloc[0]
                            result += f"- **{name}** ({designation}, {department})\n"
                        result += "\n"
            
                if not target_faculty:  # Only show general list if not asking about specific faculty
                    if teaching_at_time:
                        result += f"[TEACHING] **Faculty who are TEACHING at {target_time} on {target_day}:**\n"
                        for name in teaching_at_time:
                            faculty_row = self.faculty_data[self.faculty_data['name'] == name]
                            department = faculty_row['department'].iloc[0]
                            designation = faculty_row['designation'].iloc[0]
                            result += f"- **{name}** ({designation}, {department})\n"
                            
                            # Show what they're teaching at that time
                            for faculty_info in teaching_faculty:
                                if faculty_info['name'] == name:
                                    for i, (start, end) in enumerate(zip(faculty_info['start_times'], faculty_info['end_times'])):
                                        if self._is_time_in_range(target_time, start, end):
                                            result += f"  • {faculty_info['courses'][i]} ({start}-{end})\n"
                                    break
                            result += "\n"
        else:
            if target_faculty:
                # Specific faculty on specific day
                faculty_name = target_faculty.title()
                if faculty_name in free_faculty:
                    result = f"**Yes, {faculty_name} is FREE on {target_day}**\n"
                    result += f"==========================================\n\n"
                    result += f"**Faculty Information:**\n"
                    faculty_row = self.faculty_data[self.faculty_data['name'].str.lower() == target_faculty]
                    if not faculty_row.empty:
                        result += f"- Name: {faculty_row['name'].iloc[0]}\n"
                        result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                        result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                        result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                    result += f"\n**Status:** Available for meetings, consultations, or other activities on {target_day}."
                elif faculty_name in [f['name'] for f in teaching_faculty]:
                    result = f"**No, {faculty_name} is NOT FREE on {target_day}**\n"
                    result += f"==========================================\n\n"
                    result += f"**Faculty Information:**\n"
                    faculty_row = self.faculty_data[self.faculty_data['name'].str.lower() == target_faculty]
                    if not faculty_row.empty:
                        result += f"- Name: {faculty_row['name'].iloc[0]}\n"
                        result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                        result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                        result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                    result += f"\n**Status:** Has teaching commitments on {target_day}."
                    # Show what they're teaching
                    for faculty_info in teaching_faculty:
                        if faculty_info['name'].lower() == target_faculty:
                            result += f"\n**Teaching Schedule:**\n"
                            for i, course in enumerate(faculty_info['courses']):
                                result += f"- {course} ({faculty_info['times'][i]})\n"
                            break
                elif faculty_name in not_available:
                    result = f"**No, {faculty_name} is NOT AVAILABLE on {target_day}**\n"
                    result += f"==========================================\n\n"
                    result += f"**Faculty Information:**\n"
                    faculty_row = self.faculty_data[self.faculty_data['name'].str.lower() == target_faculty]
                    if not faculty_row.empty:
                        result += f"- Name: {faculty_row['name'].iloc[0]}\n"
                        result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                        result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                        result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                    result += f"\n**Status:** Outside working hours on {target_day}."
                else:
                    result = f"**Faculty {faculty_name} not found in the system.**"
            else:
                # General day availability (original logic)
                result = f"**Faculty Availability on {target_day}**\n"
                result += f"==========================================\n\n"
                
                # Summary
                result += f"**Summary:**\n"
                result += f"- Total Faculty: {len(self.faculty_data)}\n"
                result += f"- Free Faculty: {len(free_faculty)}\n"
                result += f"- Teaching Faculty: {len(teaching_faculty)}\n"
                result += f"- Total Classes: {sum(len(f['courses']) for f in teaching_faculty)}\n\n"
            
            if free_faculty:
                result += f"[FREE] **Faculty who are FREE on {target_day}:**\n"
                for name in free_faculty:
                    faculty_row = self.faculty_data[self.faculty_data['name'] == name]
                    department = faculty_row['department'].iloc[0]
                    designation = faculty_row['designation'].iloc[0]
                    result += f"- **{name}** ({designation}, {department})\n"
                result += "\n"
            
            if teaching_faculty:
                result += f"[TEACHING] **Faculty who are TEACHING on {target_day}:**\n"
                for faculty in teaching_faculty:
                    faculty_row = self.faculty_data[self.faculty_data['name'] == faculty['name']]
                    department = faculty_row['department'].iloc[0]
                    designation = faculty_row['designation'].iloc[0]
                    result += f"- **{faculty['name']}** ({designation}, {department})\n"
                    for i, course in enumerate(faculty['courses']):
                        result += f"  • {course} ({faculty['times'][i]})\n"
                    result += "\n"
            
            if not_available:
                result += f"[NOT AVAILABLE] **Faculty outside working hours at {target_time} on {target_day}:**\n"
                for name in not_available:
                    faculty_row = self.faculty_data[self.faculty_data['name'] == name]
                    department = faculty_row['department'].iloc[0]
                    designation = faculty_row['designation'].iloc[0]
                    availability = faculty_row['availability'].iloc[0]
                    result += f"- **{name}** ({designation}, {department}) - Available: {availability}\n"
                result += "\n"
        
        if not free_faculty and not teaching_faculty and not not_available:
            result += f"[ERROR] No faculty data available for {target_day}."
        
        return result
    
    def _is_time_in_range(self, check_time, start_time, end_time):
        """Check if a time falls within a time range"""
        try:
            check = datetime.strptime(check_time, '%H:%M').time()
            start = datetime.strptime(start_time, '%H:%M').time()
            end = datetime.strptime(end_time, '%H:%M').time()
            return start <= check < end
        except:
            return False
    
    def _check_general_availability(self, availability_str, target_day, target_time):
        """Check if faculty is generally available on the target day and time"""
        try:
            # Parse availability string like "Mon-Fri 9AM-5PM" or "Mon-Fri 8AM-4PM"
            availability_lower = availability_str.lower()
            
            # Check if target day is in availability
            day_abbrev = target_day[:3].lower()  # Mon, Tue, Wed, Thu, Fri
            if 'mon-fri' in availability_lower or 'monday-friday' in availability_lower:
                day_available = True
            elif day_abbrev in availability_lower:
                day_available = True
            else:
                day_available = False
            
            if not day_available:
                return False
            
            # If no specific time requested, just check day availability
            if target_time is None:
                return True
            
            # Extract time range from availability string
            import re
            time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?\s*-\s*(\d{1,2}):?(\d{2})?\s*(am|pm)?', availability_lower)
            if time_match:
                start_hour = int(time_match.group(1))
                start_min = int(time_match.group(2)) if time_match.group(2) else 0
                start_ampm = time_match.group(3)
                end_hour = int(time_match.group(4))
                end_min = int(time_match.group(5)) if time_match.group(5) else 0
                end_ampm = time_match.group(6)
                
                # Convert to 24-hour format
                if start_ampm == 'pm' and start_hour != 12:
                    start_hour += 12
                elif start_ampm == 'am' and start_hour == 12:
                    start_hour = 0
                    
                if end_ampm == 'pm' and end_hour != 12:
                    end_hour += 12
                elif end_ampm == 'am' and end_hour == 12:
                    end_hour = 0
                
                # Check if target time is within availability range
                target_hour = int(target_time.split(':')[0])
                target_min = int(target_time.split(':')[1])
                
                start_time_minutes = start_hour * 60 + start_min
                end_time_minutes = end_hour * 60 + end_min
                target_time_minutes = target_hour * 60 + target_min
                
                return start_time_minutes <= target_time_minutes < end_time_minutes
            
            return True  # If can't parse time, assume available
            
        except Exception as e:
            return True  # If error parsing, assume available
    
    def _get_comparison_info(self, question: str, context: str, question_info: dict) -> str:
        """Get comparison information (highest/lowest)"""
        if question_info['comparison_type'] == 'highest':
            if question_info['department']:
                return self._get_highest_workload_info(question, context)
            else:
                return self._get_highest_workload_info(question, context)
        elif question_info['comparison_type'] == 'lowest':
            if question_info['department']:
                return self._get_lowest_workload_info(question, context)
            else:
                return self._get_lowest_workload_info(question, context)
        else:
            return self._get_comprehensive_answer(question, context, question_info)
    
    def _get_listing_info(self, question: str, context: str, question_info: dict) -> str:
        """Get listing information"""
        if question_info['department']:
            return self._get_show_all_info(question, context)
        else:
            return self._get_show_all_info(question, context)
    
    def _get_statistics_info(self, question: str, context: str, question_info: dict) -> str:
        """Get statistics information"""
        question_lower = question.lower()
        
        if 'total' in question_lower or 'sum' in question_lower:
            total_hours = self.timetable_data['credits'].sum()
            total_courses = len(self.timetable_data)
            total_faculty = len(self.faculty_data)
            
            result = f"Overall Statistics:\n"
            result += f"- Total Faculty: {total_faculty}\n"
            result += f"- Total Courses: {total_courses}\n"
            result += f"- Total Credit Hours: {total_hours}\n"
            
            if question_info['department']:
                dept_faculty = self.faculty_data[self.faculty_data['department'] == question_info['department']]
                dept_courses = self.timetable_data[self.timetable_data['faculty_id'].isin(dept_faculty['faculty_id'])]
                dept_hours = dept_courses['credits'].sum()
                
                result += f"\n{question_info['department']} Department Statistics:\n"
                result += f"- Faculty: {len(dept_faculty)}\n"
                result += f"- Courses: {len(dept_courses)}\n"
                result += f"- Credit Hours: {dept_hours}\n"
            
            return result
        
        elif 'count' in question_lower or 'how many' in question_lower:
            if question_info['department']:
                dept_faculty = self.faculty_data[self.faculty_data['department'] == question_info['department']]
                return f"Number of faculty in {question_info['department']}: {len(dept_faculty)}"
            elif 'faculty' in question_lower:
                return f"Total number of faculty: {len(self.faculty_data)}"
            elif 'courses' in question_lower:
                return f"Total number of courses: {len(self.timetable_data)}"
            else:
                return f"Total number of faculty: {len(self.faculty_data)}"
        
        elif 'average' in question_lower:
            total_hours = self.timetable_data['credits'].sum()
            total_faculty = len(self.faculty_data)
            avg_hours = total_hours / total_faculty if total_faculty > 0 else 0
            
            result = f"**Average Workload Statistics**\n"
            result += f"==========================================\n\n"
            result += f"- Average workload per faculty: {avg_hours:.1f} credit hours\n"
            result += f"- Total faculty: {total_faculty}\n"
            result += f"- Total credit hours: {total_hours}\n"
            
            return result
        
        return self._get_comprehensive_answer(question, context, question_info)
    
    def _get_search_info(self, question: str, context: str, question_info: dict) -> str:
        """Get search information"""
        if question_info['faculty']:
            # Search for specific faculty
            faculty_name = question_info['faculty'][0]
            faculty_row = self.faculty_data[self.faculty_data['name'] == faculty_name]
            
            if not faculty_row.empty:
                faculty_id = faculty_row['faculty_id'].iloc[0]
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                
                result = f"Information about {faculty_name}:\n"
                result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                result += f"- Specialization: {faculty_row['specialization'].iloc[0]}\n"
                result += f"- Availability: {faculty_row['availability'].iloc[0]}\n"
                result += f"- Courses: {len(courses)}\n"
                result += f"- Credit Hours: {courses['credits'].sum()}\n"
                
                if not courses.empty:
                    result += f"\nTeaching Schedule:\n"
                    for _, course in courses.iterrows():
                        result += f"- {course['day']}: {course['course_name']} ({course['start_time']}-{course['end_time']}) in {course['room']}\n"
                
                return result
            else:
                return f"Faculty {faculty_name} not found."
        
        elif question_info['department']:
            # Search for department information
            dept_faculty = self.faculty_data[self.faculty_data['department'] == question_info['department']]
            result = f"Faculty in {question_info['department']} Department:\n"
            
            for _, faculty in dept_faculty.iterrows():
                faculty_id = faculty['faculty_id']
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                result += f"- {faculty['name']} ({faculty['designation']}): {courses['credits'].sum()} credit hours\n"
            
            return result
        
        return self._get_comprehensive_answer(question, context, question_info)
    
    def _get_comprehensive_answer(self, question: str, context: str, question_info: dict) -> str:
        """Get comprehensive answer for any question"""
        question_lower = question.lower()
        
        # Try to provide a meaningful answer based on available data
        if question_info['faculty']:
            # Answer about specific faculty
            faculty_name = question_info['faculty'][0]
            faculty_row = self.faculty_data[self.faculty_data['name'] == faculty_name]
            
            if not faculty_row.empty:
                faculty_id = faculty_row['faculty_id'].iloc[0]
                courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                
                result = f"About {faculty_name}:\n"
                result += f"- Department: {faculty_row['department'].iloc[0]}\n"
                result += f"- Designation: {faculty_row['designation'].iloc[0]}\n"
                result += f"- Specialization: {faculty_row['specialization'].iloc[0]}\n"
                result += f"- Teaching {len(courses)} courses with {courses['credits'].sum()} credit hours\n"
                
                if not courses.empty:
                    result += f"\nCourses taught:\n"
                    for _, course in courses.iterrows():
                        result += f"- {course['course_name']} on {course['day']} at {course['start_time']}\n"
                
                return result
        
        elif question_info['department']:
            # Answer about department
            dept_faculty = self.faculty_data[self.faculty_data['department'] == question_info['department']]
            dept_courses = self.timetable_data[self.timetable_data['faculty_id'].isin(dept_faculty['faculty_id'])]
            
            result = f"About {question_info['department']} Department:\n"
            result += f"- Faculty: {len(dept_faculty)}\n"
            result += f"- Courses: {len(dept_courses)}\n"
            result += f"- Total Credit Hours: {dept_courses['credits'].sum()}\n"
            
            result += f"\nFaculty members:\n"
            for _, faculty in dept_faculty.iterrows():
                result += f"- {faculty['name']} ({faculty['designation']})\n"
            
            return result
        
        elif question_info['day']:
            # Answer about specific day
            day_courses = self.timetable_data[self.timetable_data['day'] == question_info['day']]
            
            result = f"Schedule for {question_info['day']}:\n"
            if not day_courses.empty:
                for _, course in day_courses.iterrows():
                    faculty_name = self.faculty_data[self.faculty_data['faculty_id'] == course['faculty_id']]['name'].iloc[0]
                    result += f"- {course['start_time']}-{course['end_time']}: {course['course_name']} by {faculty_name} in {course['room']}\n"
            else:
                result += "No classes scheduled on this day."
            
            return result
        
        else:
            # General answer with available data
            return f"Based on the available data:\n\n{context}\n\nPlease ask a more specific question about faculty schedules, workload, departments, or timetable information."
    
    def _get_general_info(self, question: str, context: str) -> str:
        """Get general information"""
        return f"Based on the available data:\n\n{context}\n\nPlease ask a more specific question about faculty schedules, workload, or timetable conflicts."
    
    def _get_course_search_info(self, question: str, context: str) -> str:
        """Search for faculty teaching specific courses"""
        question_lower = question.lower()
        
        # Search for course names in the question
        course_names = self.timetable_data['course_name'].str.lower().tolist()
        found_courses = []
        
        for course in course_names:
            if course in question_lower:
                found_courses.append(course)
        
        if found_courses:
            result = f"**Course Teaching Information**\n"
            result += f"==========================================\n\n"
            
            for course in found_courses:
                course_data = self.timetable_data[self.timetable_data['course_name'].str.lower() == course]
                if not course_data.empty:
                    faculty_id = course_data['faculty_id'].iloc[0]
                    faculty_info = self.faculty_data[self.faculty_data['faculty_id'] == faculty_id]
                    
                    if not faculty_info.empty:
                        faculty_name = faculty_info['name'].iloc[0]
                        department = faculty_info['department'].iloc[0]
                        designation = faculty_info['designation'].iloc[0]
                        
                        result += f"**{course.title()}**\n"
                        result += f"- Taught by: {faculty_name} ({designation}, {department})\n"
                        result += f"- Day: {course_data['day'].iloc[0]}\n"
                        result += f"- Time: {course_data['start_time'].iloc[0]} - {course_data['end_time'].iloc[0]}\n"
                        result += f"- Room: {course_data['room'].iloc[0]}\n"
                        result += f"- Credits: {course_data['credits'].iloc[0]}\n\n"
            
            return result
        else:
            return "No specific course found in the question. Please mention a course name like 'Machine Learning', 'Data Structures', etc."
    
    def _get_department_workload_info(self, question: str, context: str) -> str:
        """Get department workload information"""
        question_lower = question.lower()
        
        # Extract department from question
        department = None
        if 'mathematics' in question_lower or 'math' in question_lower:
            department = 'Mathematics'
        elif 'computer science' in question_lower or 'cs' in question_lower:
            department = 'Computer Science'
        elif 'physics' in question_lower:
            department = 'Physics'
        
        if department:
            # Get faculty in the department
            dept_faculty = self.faculty_data[self.faculty_data['department'] == department]
            
            if not dept_faculty.empty:
                result = f"**{department} Department Workload Summary**\n"
                result += f"==========================================\n\n"
                
                total_hours = 0
                total_courses = 0
                
                for _, faculty in dept_faculty.iterrows():
                    faculty_id = faculty['faculty_id']
                    courses = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
                    
                    if not courses.empty:
                        faculty_hours = courses['credits'].sum()
                        faculty_courses = len(courses)
                        total_hours += faculty_hours
                        total_courses += faculty_courses
                        
                        result += f"**{faculty['name']}** ({faculty['designation']})\n"
                        result += f"- Courses: {faculty_courses}\n"
                        result += f"- Credit Hours: {faculty_hours}\n"
                        result += f"- Specialization: {faculty['specialization']}\n\n"
                
                result += f"**Department Totals:**\n"
                result += f"- Total Faculty: {len(dept_faculty)}\n"
                result += f"- Total Courses: {total_courses}\n"
                result += f"- Total Credit Hours: {total_hours}\n"
                result += f"- Average per Faculty: {total_hours/len(dept_faculty):.1f} hours\n"
                
                return result
            else:
                return f"No faculty found in {department} department."
        else:
            return "Please specify a department (Mathematics, Computer Science, or Physics) to get workload information."
    
    def get_faculty_workload(self, faculty_id: str = None) -> Dict[str, Any]:
        """Get faculty workload summary"""
        try:
            if faculty_id:
                faculty_info = self.faculty_data[self.faculty_data['faculty_id'] == faculty_id]
                timetable_info = self.timetable_data[self.timetable_data['faculty_id'] == faculty_id]
            else:
                faculty_info = self.faculty_data
                timetable_info = self.timetable_data
            
            workload = {
                'total_faculty': len(faculty_info),
                'total_courses': len(timetable_info),
                'total_hours': timetable_info['credits'].sum() if not timetable_info.empty else 0,
                'faculty_details': faculty_info.to_dict('records'),
                'timetable_details': timetable_info.to_dict('records')
            }
            
            return workload
        except Exception as e:
            return {"error": str(e)}
    
    def suggest_timetable_adjustments(self) -> List[str]:
        """Suggest timetable adjustments to avoid clashes"""
        suggestions = []
        
        try:
            # Check for time conflicts
            for day in self.timetable_data['day'].unique():
                day_schedule = self.timetable_data[self.timetable_data['day'] == day]
                
                for i, row1 in day_schedule.iterrows():
                    for j, row2 in day_schedule.iterrows():
                        if i != j and row1['faculty_id'] == row2['faculty_id']:
                            # Check for time overlap
                            start1 = datetime.strptime(row1['start_time'], '%H:%M')
                            end1 = datetime.strptime(row1['end_time'], '%H:%M')
                            start2 = datetime.strptime(row2['start_time'], '%H:%M')
                            end2 = datetime.strptime(row2['end_time'], '%H:%M')
                            
                            if (start1 < end2 and start2 < end1):
                                suggestions.append(
                                    f"Time conflict detected for {row1['faculty_id']} on {day}: "
                                    f"{row1['course_name']} ({row1['start_time']}-{row1['end_time']}) "
                                    f"overlaps with {row2['course_name']} ({row2['start_time']}-{row2['end_time']})"
                                )
            
            if not suggestions:
                suggestions.append("No time conflicts detected in the current timetable.")
                
        except Exception as e:
            suggestions.append(f"Error analyzing timetable: {str(e)}")
        
        return suggestions

def main():
    st.title("🎓 Faculty Workload & Timetable Assistant")
    st.markdown("---")
    
    # Clean sidebar without system status
    
    # Initialize the assistant
    if 'assistant' not in st.session_state:
        with st.spinner("Initializing the assistant..."):
            try:
                st.session_state.assistant = FacultyWorkloadAssistant()
                
                # Load data
                if not st.session_state.assistant.load_data():
                    st.error("Failed to load data. Please check the data files.")
                    return
                
                # Setup vector store
                if not st.session_state.assistant.setup_vectorstore():
                    st.error("Failed to setup vector store.")
                    return
                
                # Setup LLM
                if not st.session_state.assistant.setup_llm():
                    st.error("Failed to setup LLM.")
                    return
                
                # Setup QA chain
                if not st.session_state.assistant.setup_qa_chain():
                    st.error("Failed to setup QA chain.")
                    return
                
                st.success("Assistant initialized successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                logger.error(f"Initialization error: {str(e)}")
                return
    
    # CSV Upload Section
    st.sidebar.header("📁 Upload Your Data")
    st.sidebar.markdown("Upload your own faculty and timetable CSV files")
    
    # Faculty data upload
    st.sidebar.subheader("Faculty Data")
    faculty_file = st.sidebar.file_uploader(
        "Upload Faculty CSV", 
        type=['csv'],
        help="Upload faculty data with columns: faculty_id, name, department, designation, max_hours_per_week, specialization, availability"
    )
    
    # Timetable data upload
    st.sidebar.subheader("Timetable Data")
    timetable_file = st.sidebar.file_uploader(
        "Upload Timetable CSV", 
        type=['csv'],
        help="Upload timetable data with columns: course_id, course_name, faculty_id, day, start_time, end_time, room, credits"
    )
    
    # Process uploaded files
    if faculty_file is not None or timetable_file is not None:
        if st.sidebar.button("🔄 Process Uploaded Data"):
            with st.spinner("Processing uploaded data..."):
                faculty_df = None
                timetable_df = None
                
                # Process faculty file
                if faculty_file is not None:
                    faculty_df, faculty_message = st.session_state.assistant.process_uploaded_csv(faculty_file, "faculty")
                    if faculty_df is not None:
                        st.sidebar.success(f"✅ Faculty data: {faculty_message}")
                    else:
                        st.sidebar.error(f"❌ Faculty data: {faculty_message}")
                
                # Process timetable file
                if timetable_file is not None:
                    timetable_df, timetable_message = st.session_state.assistant.process_uploaded_csv(timetable_file, "timetable")
                    if timetable_df is not None:
                        st.sidebar.success(f"✅ Timetable data: {timetable_message}")
                    else:
                        st.sidebar.error(f"❌ Timetable data: {timetable_message}")
                
                # Reload data with uploaded files
                if faculty_df is not None or timetable_df is not None:
                    st.session_state.assistant.load_data(faculty_df, timetable_df)
                    st.session_state.assistant.setup_vectorstore()
                    st.session_state.assistant.setup_qa_chain()
                    st.sidebar.success("🎉 Data updated successfully!")
                    st.rerun()
    
    # Show current data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Data")
    if st.session_state.assistant.faculty_data is not None:
        st.sidebar.write(f"👥 Faculty: {len(st.session_state.assistant.faculty_data)}")
    if st.session_state.assistant.timetable_data is not None:
        st.sidebar.write(f"📚 Courses: {len(st.session_state.assistant.timetable_data)}")
    
    # Download sample files
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Sample Files")
    
    if st.sidebar.button("Download Faculty Sample"):
        sample_faculty = st.session_state.assistant.create_sample_faculty_data()
        csv = sample_faculty.to_csv(index=False)
        st.sidebar.download_button(
            label="Download faculty_data.csv",
            data=csv,
            file_name="faculty_data.csv",
            mime="text/csv"
        )
    
    if st.sidebar.button("Download Timetable Sample"):
        sample_timetable = st.session_state.assistant.create_sample_timetable_data()
        csv = sample_timetable.to_csv(index=False)
        st.sidebar.download_button(
            label="Download timetable_data.csv",
            data=csv,
            file_name="timetable_data.csv",
            mime="text/csv"
        )
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "👥 Faculty Workload", "📅 Timetable Analysis", "📊 Data Overview"])
    
    with tab1:
        st.header("Ask Questions About Faculty & Timetables")
        
        question = st.text_input("Enter your question:", placeholder="e.g., What is Dr. John Smith's schedule?")
        
        if st.button("Ask Question"):
            if question:
                with st.spinner("Thinking..."):
                    answer = st.session_state.assistant.answer_question(question)
                    st.write("**Answer:**")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")
    
    with tab2:
        st.header("Faculty Workload Summary")
        
        faculty_id = st.selectbox(
            "Select Faculty:",
            ["All"] + list(st.session_state.assistant.faculty_data['faculty_id'].unique())
        )
        
        if st.button("Get Workload Summary"):
            if faculty_id == "All":
                workload = st.session_state.assistant.get_faculty_workload()
            else:
                workload = st.session_state.assistant.get_faculty_workload(faculty_id)
            
            if 'error' not in workload:
                st.write(f"**Total Faculty:** {workload['total_faculty']}")
                st.write(f"**Total Courses:** {workload['total_courses']}")
                st.write(f"**Total Hours:** {workload['total_hours']}")
                
                st.subheader("Faculty Details")
                st.dataframe(pd.DataFrame(workload['faculty_details']))
                
                st.subheader("Timetable Details")
                st.dataframe(pd.DataFrame(workload['timetable_details']))
            else:
                st.error(workload['error'])
    
    with tab3:
        st.header("Timetable Analysis & Suggestions")
        
        if st.button("Analyze Timetable"):
            suggestions = st.session_state.assistant.suggest_timetable_adjustments()
            
            st.subheader("Timetable Analysis Results")
            for suggestion in suggestions:
                if "conflict" in suggestion.lower():
                    st.error(suggestion)
                else:
                    st.success(suggestion)
    
    with tab4:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Faculty Data")
            st.dataframe(st.session_state.assistant.faculty_data)
        
        with col2:
            st.subheader("Timetable Data")
            st.dataframe(st.session_state.assistant.timetable_data)

if __name__ == "__main__":
    main()

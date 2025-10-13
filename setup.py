#!/usr/bin/env python3
"""
Setup script for Faculty Workload & Timetable Assistant
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating necessary directories...")
    directories = ["data", "chroma_db"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def main():
    """Main setup function"""
    print("Setting up Faculty Workload & Timetable Assistant...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        print("\nSetup completed successfully!")
        print("\nTo run the application, use:")
        print("streamlit run app.py")
    else:
        print("\nSetup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

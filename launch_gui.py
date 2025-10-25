#!/usr/bin/env python3
"""
LLM-Feynman GUI Launcher Script
"""

import os
import sys

# Set environment variables for English interface (only analytics, avoid locale issues)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """Main function"""
    try:
        print("🚀 Starting LLM-Feynman GUI...")
        print("📍 Current working directory:", current_dir)
        
        # Check dependencies
        try:
            import gradio as gr
            print("✅ Gradio is installed")
        except ImportError:
            print("❌ Gradio is not installed, please run: pip install gradio")
            return
        
        try:
            import pandas as pd
            print("✅ Pandas is installed")
        except ImportError:
            print("❌ Pandas is not installed, please run: pip install pandas")
            return
        
        try:
            import plotly
            print("✅ Plotly is installed")
        except ImportError:
            print("❌ Plotly is not installed, please run: pip install plotly")
            return
        
        # Import GUI
        from gradio_gui import create_gui
        
        # Create and launch interface
        interface = create_gui()
        
        print("🌐 Starting Web interface...")
        print("🔗 Access URL: http://localhost:7860")
        print("📝 Instructions: Please refer to the usage instructions at the bottom of the interface")
        print("⚠️  For first time use, it is recommended to use test_data_ohms_law.csv as test data")
        print("✅ GUI started successfully, you can access it in the browser!")
        
        interface.launch(
            server_name="127.0.0.1",  # Only allow local access
            server_port=7860,
            share=False,
            show_error=True,
            debug=False,  # Reduce debug output
            quiet=False   # Show startup information
        )
        
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
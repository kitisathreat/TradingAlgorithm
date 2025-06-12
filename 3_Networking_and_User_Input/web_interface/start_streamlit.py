import subprocess
import sys
import webbrowser
import time
import os

def start_streamlit():
    print("Starting Trading Algorithm Interface...")
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Start the Streamlit app
        process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Open the browser
        webbrowser.open('http://localhost:8501')
        
        print("✨ Trading Algorithm Interface is running!")
        print("📊 Access the interface at: http://localhost:8501")
        print("Press Ctrl+C to stop the server...")
        
        # Keep the script running and display any output
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            if process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        print("\nStopping the server...")
        process.terminate()
        process.wait()
        print("Server stopped successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'process' in locals():
            process.terminate()
            process.wait()

if __name__ == "__main__":
    start_streamlit() 
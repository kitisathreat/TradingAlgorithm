import subprocess
import webbrowser
import time
import sys
import os
import logging
import threading
import tkinter as tk
from tkinter import ttk

# Try to import tqdm, install if missing
try:
    from tqdm import tqdm
except ImportError:
    import subprocess as sp
    print("tqdm not found, installing...")
    sp.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressPopup:
    def __init__(self, root):
        self.root = root
        self.root.title("Server & Browser Startup Progress")
        self.root.geometry("400x220")
        self.root.resizable(False, False)

        self.status_label = ttk.Label(root, text="Initializing...", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.server_bar = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate", maximum=20)
        self.server_bar.pack(pady=10)
        self.server_label = ttk.Label(root, text="Server startup progress")
        self.server_label.pack()

        self.browser_bar = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate", maximum=10)
        self.browser_bar.pack(pady=10)
        self.browser_label = ttk.Label(root, text="Browser launch progress")
        self.browser_label.pack()

        self.close_button = ttk.Button(root, text="Close", command=self.root.destroy, state=tk.DISABLED)
        self.close_button.pack(pady=10)

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def update_server_bar(self, value):
        self.server_bar['value'] = value
        self.root.update_idletasks()

    def update_browser_bar(self, value):
        self.browser_bar['value'] = value
        self.root.update_idletasks()

    def enable_close(self):
        self.close_button.config(state=tk.NORMAL)
        self.root.update_idletasks()

def launch_server_and_browser(popup):
    try:
        # Change to the API directory
        api_dir = os.path.join(os.path.dirname(__file__), 'api')
        os.chdir(api_dir)
        logger.info(f"Changed to directory: {api_dir}")
        popup.update_status("Starting FastAPI server...")

        # Start the FastAPI server in a subprocess
        server = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'main:app', '--reload'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Progress bar for server startup
        server_started = False
        for i in range(20):
            time.sleep(0.25)
            popup.update_server_bar(i+1)
            popup.update_status(f"Waiting for server to start... ({int((i+1)/20*100)}%)")
            if server.poll() is not None:
                break
            if server.stderr:
                line = server.stderr.readline()
                if line and ("Uvicorn running" in line or "Application startup complete" in line):
                    server_started = True
                    break
        if not server_started:
            if server.poll() is not None:
                stdout, stderr = server.communicate()
                logger.error(f"Server failed to start. Error: {stderr}")
                popup.update_status("[X] Server failed to start. See error log.")
                popup.enable_close()
                return
            else:
                popup.update_status("[!] Server is taking longer than expected to start. It may be hanging.")
                # Optionally, continue to try launching browser anyway
        else:
            popup.update_status("[✓] Server started successfully!")

        # Progress bar for browser launch
        for i in range(10):
            time.sleep(0.1)
            popup.update_browser_bar(i+1)
            popup.update_status(f"Launching browser... ({int((i+1)/10*100)}%)")
        webbrowser.open('http://localhost:8000')
        popup.update_status("[✓] Browser launched. If the page does not load, check the server logs.")
        popup.enable_close()

        # Optionally, monitor server output in the background (not blocking GUI)
        def monitor_server():
            while True:
                output = server.stderr.readline()
                if output:
                    logger.info(f"Server: {output.strip()}")
                if server.poll() is not None:
                    logger.info("[!] Server process exited.")
                    break
        threading.Thread(target=monitor_server, daemon=True).start()

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        popup.update_status(f"[X] Error: {str(e)}")
        popup.enable_close()

if __name__ == "__main__":
    root = tk.Tk()
    popup = ProgressPopup(root)
    threading.Thread(target=launch_server_and_browser, args=(popup,), daemon=True).start()
    root.mainloop() 
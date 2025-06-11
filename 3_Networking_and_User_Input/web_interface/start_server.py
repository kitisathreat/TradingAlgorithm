import subprocess
import webbrowser
import time
import sys
import os
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import traceback
import socket

# Try to import tqdm, install if missing
try:
    from tqdm import tqdm
except ImportError:
    import subprocess as sp
    print("tqdm not found, installing...")
    sp.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    from tqdm import tqdm

# Get the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging to also write to a file
log_file = os.path.join(SCRIPT_DIR, 'server_startup.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 'w' mode to clear previous logs
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== Server Startup Script Started ===")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Current working directory: {os.getcwd()}")

class ProgressPopup:
    def __init__(self, root):
        self.root = root
        self.root.title("Server & Browser Startup Progress")
        self.root.geometry("600x300")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Always allow closing
        self.server_process = None  # Store server process reference
        self.stopped = False  # Track if force stop was used

        # Create a frame for better padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add a text widget for detailed logging with scrollbar
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=6, width=70, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_label = ttk.Label(main_frame, text="Initializing...", font=("Arial", 12), wraplength=580)
        self.status_label.pack(pady=5, fill=tk.X)

        self.server_bar = ttk.Progressbar(main_frame, orient="horizontal", length=550, mode="determinate", maximum=100)
        self.server_bar.pack(pady=5, fill=tk.X)
        self.server_label = ttk.Label(main_frame, text="Server startup progress")
        self.server_label.pack()

        self.browser_bar = ttk.Progressbar(main_frame, orient="horizontal", length=550, mode="determinate", maximum=100)
        self.browser_bar.pack(pady=5, fill=tk.X)
        self.browser_label = ttk.Label(main_frame, text="Browser launch progress")
        self.browser_label.pack()

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=5, fill=tk.X)

        self.close_button = ttk.Button(button_frame, text="Close", command=self.on_closing)
        self.close_button.pack(side=tk.RIGHT, padx=5)

        self.force_stop_button = ttk.Button(button_frame, text="Force Stop", command=self.force_stop)
        self.force_stop_button.pack(side=tk.RIGHT, padx=5)

        self.debug_button = ttk.Button(button_frame, text="Debug Info", command=self.show_debug_info)
        self.debug_button.pack(side=tk.RIGHT, padx=5)

        self.launch_browser_button = ttk.Button(button_frame, text="Launch Browser", command=self.launch_browser, state=tk.DISABLED)
        self.launch_browser_button.pack(side=tk.RIGHT, padx=5)

        # Schedule regular UI updates
        self.schedule_updates()

    def on_closing(self):
        self.force_stop()  # Always clean up server process
        self.root.destroy()

    def force_stop(self):
        if self.server_process and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.log_message("[!] Server process forcibly stopped.")
        self.stopped = True
        self.status_label.config(text="[!] Force stopped. You may close this window.")
        self.server_bar['value'] = 0
        self.browser_bar['value'] = 0
        self.launch_browser_button.config(state=tk.DISABLED)
        self.force_stop_button.config(state=tk.DISABLED)

    def launch_browser(self):
        try:
            self.update_status("Launching browser...")
            for i in range(101):
                time.sleep(0.02)
                self.update_browser_bar(i)
                self.root.update_idletasks()
            webbrowser.open('http://localhost:8000')
            self.update_browser_bar(100)
            self.update_status("[✓] Browser launched successfully")
        except Exception as e:
            self.log_message(f"Error launching browser: {str(e)}")
            self.update_status("[!] Failed to launch browser. Please open http://localhost:8000 manually.")

    def schedule_updates(self):
        self.root.update()
        self.root.after(100, self.schedule_updates)

    def show_debug_info(self):
        """Show debug information in the log window"""
        self.log_message("\n=== Debug Information ===")
        self.log_message(f"Current working directory: {os.getcwd()}")
        self.log_message(f"Python executable: {sys.executable}")
        self.log_message(f"Script directory: {SCRIPT_DIR}")
        self.log_message(f"API directory exists: {os.path.exists(os.path.join(SCRIPT_DIR, 'api'))}")
        self.log_message(f"main.py exists: {os.path.exists(os.path.join(SCRIPT_DIR, 'api', 'main.py'))}")
        self.log_message("=== End Debug Info ===\n")

    def log_message(self, message):
        """Add a message to the log window and also log it to file"""
        logger.info(message)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.log_message(msg)
        self.root.update_idletasks()

    def update_server_bar(self, value):
        self.server_bar['value'] = value
        self.root.update_idletasks()

    def update_browser_bar(self, value):
        self.browser_bar['value'] = value
        self.root.update_idletasks()

def check_port_in_use(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return False
    except socket.error:
        return True
    except Exception as e:
        logger.error(f"Error checking port: {e}")
        return False

def launch_server_and_browser(popup):
    try:
        popup.log_message("Starting server launch process...")
        
        # Check if port 8000 is already in use using a more reliable method
        if check_port_in_use(8000):
            popup.update_status("[!] Port 8000 is already in use. Please close any existing server processes.")
            # Try to find and kill any process using port 8000
            try:
                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    try:
                        for conn in proc.connections():
                            if conn.laddr.port == 8000:
                                popup.log_message(f"Found process using port 8000: {proc.name()} (PID: {proc.pid})")
                                proc.kill()
                                popup.log_message(f"Killed process {proc.pid}")
                                time.sleep(1)  # Wait a bit for the port to be released
                                if not check_port_in_use(8000):
                                    popup.log_message("Port 8000 is now free")
                                    break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
            except Exception as e:
                popup.log_message(f"Error trying to free port 8000: {str(e)}")
                return
            else:
                if check_port_in_use(8000):
                    popup.update_status("[!] Could not free port 8000. Please restart your computer and try again.")
                    return

        # Change to the API directory using absolute path
        api_dir = os.path.join(SCRIPT_DIR, 'api')
        popup.log_message(f"API directory path: {api_dir}")
        
        if not os.path.exists(api_dir):
            popup.update_status(f"[X] API directory not found at: {api_dir}")
            return

        try:
            os.chdir(api_dir)
            popup.log_message(f"Successfully changed to directory: {api_dir}")
        except Exception as e:
            popup.update_status(f"[X] Failed to change to API directory: {str(e)}")
            return

        popup.log_message(f"Current working directory: {os.getcwd()}")
        popup.log_message(f"Directory contents: {os.listdir('.')}")

        # Check if main.py exists
        if not os.path.exists('main.py'):
            popup.update_status("[X] main.py not found in the API directory")
            return

        popup.update_status("Starting FastAPI server...")

        # Start the FastAPI server in a subprocess with more debugging
        env = os.environ.copy()
        env["PYTHONPATH"] = SCRIPT_DIR
        env["PYTHONUNBUFFERED"] = "1"  # Force Python to run unbuffered
        python_exe = sys.executable
        
        popup.log_message(f"Starting server with Python: {python_exe}")
        popup.log_message(f"PYTHONPATH set to: {env['PYTHONPATH']}")
        
        try:
            # Start server using a different approach - direct uvicorn command without FastAPI's dev server
            server = subprocess.Popen(
                [python_exe, '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000', '--reload-dir', api_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=api_dir,
                creationflags=subprocess.CREATE_NO_WINDOW  # Prevent console window from appearing
            )
            popup.server_process = server  # Store reference to server process
            popup.log_message("Server process started successfully")
            
            # Immediately try to read any initial output
            for pipe in [server.stdout, server.stderr]:
                if pipe:
                    try:
                        line = pipe.readline()
                        if line:
                            popup.log_message(f"Initial server output: {line.strip()}")
                    except Exception as e:
                        popup.log_message(f"Error reading initial output: {str(e)}")
            
        except Exception as e:
            popup.update_status(f"[X] Failed to start server process: {str(e)}")
            return

        # Start a separate thread for reading server output
        def read_server_output():
            server_started = False
            start_time = time.time()
            startup_timeout = 60  # 60 second timeout
            error_buffer = []
            spinner_states = ['.', '..', '...', '....']
            spinner_index = 0
            startup_patterns = [
                "Uvicorn running",
                "Application startup complete",
                "Started server process",
                "Waiting for application startup",
                "Application startup complete",
                "INFO:     Uvicorn running",
                "INFO:     Application startup complete",
                "INFO:     Started server process",
                "INFO:     Started reloader process",
                "INFO:     Started server process",
                "INFO:     Waiting for application startup",
                "INFO:     Application startup complete"
            ]

            while True:
                if server.poll() is not None:
                    stdout, stderr = server.communicate()
                    popup.log_message(f"Server process exited with code {server.returncode}")
                    if stdout:
                        popup.log_message(f"Server stdout: {stdout}")
                    if stderr:
                        popup.log_message(f"Server stderr: {stderr}")
                    if not stdout and not stderr:
                        popup.log_message("Server process exited without any output")
                    popup.update_status(f"[X] Server failed to start.\nError: {stderr if stderr else stdout or 'No output received'}")
                    return

                # Update progress and spinner
                elapsed_time = time.time() - start_time
                if elapsed_time >= startup_timeout:
                    if error_buffer:
                        popup.update_status(f"[!] Server startup timeout.\nLast errors:\n" + "\n".join(error_buffer[-5:]))
                    else:
                        popup.update_status("[!] Server startup timeout. No error messages received.")
                    return

                progress = min((elapsed_time / startup_timeout) * 100, 99)
                popup.update_server_bar(progress)
                
                spinner = spinner_states[spinner_index % len(spinner_states)]
                spinner_index += 1
                if error_buffer:
                    popup.update_status(f"Waiting for server{spinner} ({int(progress)}%)\nLast error: {error_buffer[-1]}")
                else:
                    popup.update_status(f"Waiting for server to start{spinner} ({int(progress)}%)")

                # Read server output with timeout
                for pipe in [server.stdout, server.stderr]:
                    if pipe:
                        try:
                            # Use select to implement a timeout for reading
                            import select
                            if select.select([pipe], [], [], 0.1)[0]:  # 0.1 second timeout
                                line = pipe.readline()
                                if line:
                                    line = line.strip()
                                    popup.log_message(f"Server: {line}")
                                    # Check for any error-like messages
                                    if any(err in line.lower() for err in ["error", "exception", "failed", "traceback", "not found", "cannot", "invalid", "address already in use"]):
                                        error_buffer.append(line)
                                    # Check for startup patterns
                                    if any(pattern.lower() in line.lower() for pattern in startup_patterns):
                                        server_started = True
                                        popup.log_message("Server startup detected!")
                                        popup.update_server_bar(100)
                                        popup.update_status("[✓] Server started successfully!")
                                        popup.launch_browser_button.config(state=tk.NORMAL)  # Enable browser launch button
                                        return
                        except Exception as e:
                            popup.log_message(f"Error reading server output: {str(e)}")
                            error_buffer.append(f"Error reading output: {str(e)}")

                time.sleep(0.1)

        # Start the output reading thread
        threading.Thread(target=read_server_output, daemon=True).start()

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error: {error_msg}")
        popup.update_status(f"[X] Error occurred:\n{error_msg}")

if __name__ == "__main__":
    try:
        logger.info("=== Starting main script ===")
        root = tk.Tk()
        popup = ProgressPopup(root)
        
        # Start the server in a separate thread
        server_thread = threading.Thread(target=launch_server_and_browser, args=(popup,), daemon=True)
        server_thread.start()
        
        # Start the main event loop
        root.mainloop()
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error: {error_msg}")
        print(f"Error: {error_msg}")
        input("Press Enter to exit...")
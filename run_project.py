import subprocess
import webbrowser
import time
import os
import sys

def run_project():
    print("Starting Backend Server (FastAPI)...")
    # Start Backend
    # ensuring we use the same python interpreter
    backend = subprocess.Popen([sys.executable, "main.py"], cwd=os.getcwd())
    
    print("Starting Frontend Server (HTTP)...")
    # Start Frontend on port 3000
    frontend = subprocess.Popen([sys.executable, "-m", "http.server", "3000"], cwd=os.getcwd())
    
    # Wait a few seconds for servers to initialize
    time.sleep(3)
    
    url = "http://localhost:3000/index.html"
    print(f"Opening Browser at {url}...")
    webbrowser.open(url)
    
    print("Press Ctrl+C to stop the servers.")
    
    try:
        # Keep the script running to keep subprocesses alive
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend.terminate()
        frontend.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    run_project()

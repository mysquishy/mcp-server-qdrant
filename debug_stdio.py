#!/usr/bin/env python
"""
Debug script that creates a simple stdin/stdout pipe to the MCP server
"""
import subprocess
import time
import json
import sys
import threading

def read_output(process, prefix=""):
    """Continuously read and print output from a subprocess"""
    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(f"{prefix} > {line.strip()}")
    
    print(f"{prefix} stdout closed")

def read_errors(process, prefix=""):
    """Continuously read and print errors from a subprocess"""
    for line in iter(process.stderr.readline, ""):
        if line.strip():
            print(f"{prefix} ERR > {line.strip()}")
    
    print(f"{prefix} stderr closed")

def main():
    # Start the server with extensive debugging
    cmd = ["mcp-server-qdrant"]
    
    print(f"Starting process: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Start threads to read output and errors
    stdout_thread = threading.Thread(target=read_output, args=(process, "SERVER"))
    stderr_thread = threading.Thread(target=read_errors, args=(process, "SERVER"))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Give the server some time to initialize
    time.sleep(3)
    
    # Send a basic initialization message
    init_message = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "clientName": "test-client",
            "clientVersion": "1.0.0"
        },
        "id": "1"
    }
    
    print("\nSending initialization message:")
    print(json.dumps(init_message, indent=2))
    
    try:
        process.stdin.write(json.dumps(init_message) + "\n")
        process.stdin.flush()
        print("Message sent successfully")
    except BrokenPipeError:
        print("ERROR: Could not write to process stdin (pipe is broken)")
    
    # Wait a bit for any response
    try:
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("\nProcess is still running. Press Ctrl+C to terminate.")
            
            # Keep the main thread alive until Ctrl+C
            while True:
                time.sleep(1)
        else:
            print(f"\nProcess exited with code {process.returncode}")
    
    except KeyboardInterrupt:
        print("\nTerminating process...")
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print("Process didn't terminate gracefully, killing it")
            process.kill()
    
    print("Debug session ended")

if __name__ == "__main__":
    main()

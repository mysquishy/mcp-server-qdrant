#!/usr/bin/env python
"""
Launcher script for the Enhanced MCP Server
This script ensures all dependencies are correctly on the Python path
"""
import os
import sys
import site
import platform
import subprocess

def main():
    """Main entry point for the launcher"""
    print("Enhanced MCP Server Launcher")
    print("===========================")
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()
    print(f"Python version: {platform.python_version()}")
    print(f"Site packages: {site_packages}")
    
    # Add site-packages to Python path
    for sp in site_packages:
        if sp not in sys.path:
            sys.path.insert(0, sp)
            print(f"Added {sp} to Python path")
    
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added {current_dir} to Python path")
    
    # Try importing key packages to verify they're accessible
    packages_to_check = ["sklearn", "hdbscan", "umap", "nltk"]
    missing_packages = []
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✓ Successfully imported {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ Failed to import {package}")
    
    if missing_packages:
        print("\nThe following packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them with:")
        for package in missing_packages:
            print(f"pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org {package}")
        
        user_input = input("\nDo you want to try running the server anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting.")
            return
    
    print("\nStarting the Enhanced MCP Server...")
    
    # Set PYTHONPATH environment variable
    os.environ["PYTHONPATH"] = ":".join(sys.path)
    
    # Run the enhanced MCP server
    server_path = os.path.join(current_dir, "enhanced_mcp_server.py")
    try:
        subprocess.run([sys.executable, server_path], env=os.environ)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error running server: {str(e)}")

if __name__ == "__main__":
    main()

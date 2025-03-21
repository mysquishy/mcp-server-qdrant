#!/usr/bin/env python
"""
Script to check and set up the Python environment for running the MCP server
"""
import sys
import site
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed and importable"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✓ Successfully imported {package_name} from {module.__file__}")
        return True
    except ImportError:
        print(f"✗ Failed to import {package_name}")
        return False

# Print environment info
print("\n=== Python Environment Information ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Site packages: {site.getsitepackages()}")

# Check required packages
print("\n=== Required Packages ===")
packages = ["sklearn", "hdbscan", "umap", "nltk"]
missing = []

for pkg in packages:
    if not check_package(pkg):
        missing.append(pkg)

if missing:
    print(f"\n⚠️ Missing packages: {', '.join(missing)}")
    print("\nTry installing them with:")
    for pkg in missing:
        print(f"pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org {pkg}")
else:
    print("\n✓ All required packages are installed and importable")
    print("\nYou can now run the enhanced MCP server with:")
    print("python enhanced_mcp_server.py")

# Print instructions for adding site-packages to PYTHONPATH
print("\n=== Fixing Import Issues ===")
site_packages = site.getsitepackages()[0]
print(f"If you're still having import issues, try setting PYTHONPATH to include {site_packages}:")
print(f"PYTHONPATH={site_packages} python enhanced_mcp_server.py")

"""
Cleanup script - Remove temporary files before commit

Usage:
    python scripts/cleanup.py
"""

import os
import shutil
from pathlib import Path

def cleanup_directory(directory):
    """Remove temporary files from a directory."""
    patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.pytest_cache',
        '.mypy_cache',
        '*.egg-info',
        '.DS_Store',
        'Thumbs.db',
        '*.tmp',
        '*.log',
    ]
    
    removed = []
    for pattern in patterns:
        for file in Path(directory).rglob(pattern):
            try:
                if file.is_file():
                    file.unlink()
                    removed.append(str(file))
                elif file.is_dir():
                    shutil.rmtree(file)
                    removed.append(str(file) + '/')
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    return removed

if __name__ == '__main__':
    print("Cleaning up temporary files...")
    print("-" * 50)
    
    # Clean current directory
    removed = cleanup_directory('.')
    
    if removed:
        print(f"Removed {len(removed)} items:")
        for item in removed[:20]:  # Show first 20
            print(f"  - {item}")
        if len(removed) > 20:
            print(f"  ... and {len(removed) - 20} more")
    else:
        print("No temporary files found.")
    
    print("-" * 50)
    print("Cleanup complete!")

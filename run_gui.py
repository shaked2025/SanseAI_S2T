"""
Simplified launcher that ensures GUI appears
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

def test_gui():
    """Test if GUI can be created"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide for now
        
        # Test if we can create a window
        result = messagebox.askyesno(
            "Speech-to-Text System",
            "Ready to launch the speech-to-text application?\n\n"
            "Click 'Yes' to start.\n"
            "Click 'No' to cancel."
        )
        
        root.destroy()
        return result
        
    except Exception as e:
        print(f"GUI Error: {e}")
        return False

def main():
    print("="*60)
    print(" "*15 + "GUI Launcher")
    print("="*60)
    print()
    
    if test_gui():
        print("Launching application...")
        print()
        # Run the main application
        os.system("python main.py")
    else:
        print("Launch cancelled")

if __name__ == "__main__":
    main()


"""
Force window to be visible on top
"""
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys

def show_dialog_then_launch():
    """Show a message box that WILL appear, then launch app"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    
    result = messagebox.showinfo(
        "Speech-to-Text System",
        "The application will now open.\n\n"
        "You will see:\n"
        "- Video feed on the left\n"
        "- Transcript area on the right\n"
        "- Start/Stop buttons at bottom\n\n"
        "Click OK to continue...",
        parent=root
    )
    
    root.destroy()
    
    # Now run main app
    subprocess.Popen([sys.executable, "main.py"])
    
    # Show another dialog to confirm
    root2 = tk.Tk()
    root2.withdraw()
    root2.attributes('-topmost', True)
    
    messagebox.showinfo(
        "Look for the Window",
        "The application is now running!\n\n"
        "If you don't see it:\n"
        "1. Press Alt+Tab to find it\n"
        "2. Check your taskbar\n"
        "3. Look for 'Real-Time Speech-to-Text' window",
        parent=root2
    )
    
    root2.destroy()

if __name__ == "__main__":
    show_dialog_then_launch()


"""
Simple GUI test to verify tkinter works
"""

import tkinter as tk
from tkinter import ttk
import sys

print("Creating simple test window...")

try:
    root = tk.Tk()
    root.title("GUI Test - Speech-to-Text")
    root.geometry("800x600")
    
    # Force to front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    # Add a simple label
    label = tk.Label(root, text="✓ GUI is working!\n\nIf you see this window, tkinter is functioning correctly.",
                     font=("Arial", 16), pady=50)
    label.pack()
    
    # Add instructions
    info = tk.Label(root, text="This is a test window.\nClose this window to continue.",
                   font=("Arial", 12))
    info.pack()
    
    # Add a button
    button = tk.Button(root, text="Close Test", command=root.destroy,
                      bg='#3498DB', fg='white', font=('Arial', 14), padx=20, pady=10)
    button.pack(pady=20)
    
    print("Test window created successfully!")
    print("Look for the window on your screen...")
    print("If you don't see it, check taskbar or try Alt+Tab")
    
    root.mainloop()
    
    print("Test window closed successfully!")
    
except Exception as e:
    print(f"ERROR: Failed to create test window: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


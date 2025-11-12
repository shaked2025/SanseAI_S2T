#!/bin/bash
# Unix/Linux/Mac script to run the Speech-to-Text application

echo "================================================"
echo "   Real-Time Speech-to-Text System"
echo "================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if requirements are installed
python3 -c "import whisper" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "It looks like dependencies are not installed."
    echo "Would you like to run the installer? (y/n)"
    read -r INSTALL_CHOICE
    if [ "$INSTALL_CHOICE" = "y" ] || [ "$INSTALL_CHOICE" = "Y" ]; then
        python3 install.py
        echo ""
        echo "Press Enter to start the application..."
        read -r
    else
        echo ""
        echo "Please run: python3 install.py"
        echo "Then run this script again."
        exit 1
    fi
fi

# Run the application
echo "Starting application..."
echo ""
python3 main.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error."
    echo "Press Enter to exit..."
    read -r
fi


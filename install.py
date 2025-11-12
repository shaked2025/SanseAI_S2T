"""
Installation Helper Script
Checks dependencies and helps set up the environment
"""

import sys
import subprocess
import platform


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8 or higher required. You have Python {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_pip():
    """Check if pip is available"""
    print("\nChecking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✓ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False


def install_requirements():
    """Install required packages"""
    print("\n" + "="*60)
    print("Installing required packages...")
    print("="*60)
    print("\nThis may take several minutes as it will download:")
    print("  - OpenAI Whisper model files")
    print("  - PyTorch (deep learning framework)")
    print("  - Audio/Video processing libraries")
    print()
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                      check=True)
        
        # Install PyTorch (CPU version for compatibility)
        print("\nInstalling PyTorch...")
        if platform.system() == "Windows":
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "torch", "torchaudio", "--index-url", 
                          "https://download.pytorch.org/whl/cpu"],
                         check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "torch", "torchaudio"],
                         check=True)
        
        # Install other requirements
        print("\nInstalling other dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        
        print("\n✓ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing packages: {e}")
        return False


def test_imports():
    """Test if critical imports work"""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    modules = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("pyaudio", "PyAudio"),
        ("whisper", "OpenAI Whisper"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML")
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name} - {e}")
            all_ok = False
            
    return all_ok


def check_audio_devices():
    """Check available audio input devices"""
    print("\n" + "="*60)
    print("Checking audio devices...")
    print("="*60)
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        device_count = p.get_device_count()
        print(f"\nFound {device_count} audio device(s):")
        
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (Input channels: {info['maxInputChannels']})")
                
        p.terminate()
        return True
        
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")
        return False


def check_camera():
    """Check if camera is available"""
    print("\n" + "="*60)
    print("Checking camera...")
    print("="*60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera is working")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print("❌ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera")
            return False
            
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False


def download_whisper_model():
    """Pre-download Whisper model"""
    print("\n" + "="*60)
    print("Downloading Whisper model...")
    print("="*60)
    print("\nThis will download the 'base' model (~140MB)")
    print("You can change the model size in config.yaml later")
    
    try:
        import whisper
        print("\nDownloading...")
        model = whisper.load_model("base")
        print("✓ Whisper model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def main():
    """Main installation process"""
    print("="*60)
    print(" "*15 + "Installation Helper")
    print(" "*10 + "Speech-to-Text System Setup")
    print("="*60)
    print()
    
    # Check Python version
    if not check_python_version():
        print("\nPlease upgrade Python and try again.")
        return False
        
    # Check pip
    if not check_pip():
        print("\nPlease install pip and try again.")
        return False
    
    # Ask user if they want to install
    print("\n" + "="*60)
    response = input("\nInstall required packages? (y/n): ").strip().lower()
    
    if response == 'y':
        if not install_requirements():
            print("\nInstallation failed. Please check the errors above.")
            return False
    else:
        print("\nSkipping package installation.")
        
    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed. The application may not work correctly.")
        print("Try running: pip install -r requirements.txt")
        
    # Check audio devices
    check_audio_devices()
    
    # Check camera
    check_camera()
    
    # Download Whisper model
    print("\n" + "="*60)
    response = input("\nDownload Whisper model now? (y/n): ").strip().lower()
    if response == 'y':
        download_whisper_model()
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nTo run the application:")
    print("  python main.py")
    print()
    print("Configuration:")
    print("  Edit config.yaml to customize settings")
    print()
    print("Documentation:")
    print("  See README.md for detailed instructions")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


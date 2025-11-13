"""
Quick Microphone Test
Verifies external microphone is working and shows audio levels
"""

import pyaudio
import numpy as np
import time
import sys


def test_microphone(duration=5):
    """Test current microphone for specified duration"""
    
    print("="*60)
    print("            MICROPHONE TEST")
    print("="*60)
    print()
    
    # List available input devices
    p = pyaudio.PyAudio()
    
    print("Available Audio Input Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            default = " [DEFAULT]" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  [{i}] {info['name']}{default}")
            
    print()
    
    # Get default device
    default_device = p.get_default_input_device_info()
    print(f"Currently Using: {default_device['name']}")
    print(f"Sample Rate: {int(default_device['defaultSampleRate'])} Hz")
    print(f"Channels: {default_device['maxInputChannels']}")
    print()
    
    # Open stream
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print(f"🎤 Testing microphone for {duration} seconds...")
        print("SPEAK NOW to see audio levels!")
        print()
        print("Audio Level: ", end='', flush=True)
        
        start_time = time.time()
        max_level = 0
        avg_levels = []
        
        while (time.time() - start_time) < duration:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            max_level = max(max_level, rms)
            avg_levels.append(rms)
            
            # Visual bar
            bars = int(rms / 500)  # Scale for display
            bar_str = '█' * min(bars, 50)
            
            # Print with carriage return to update same line
            print(f"\rAudio Level: {bar_str:<50} {int(rms):>5}  ", end='', flush=True)
            
        print()  # New line
        print()
        
        # Results
        avg_level = np.mean(avg_levels)
        
        print("="*60)
        print("TEST RESULTS:")
        print("="*60)
        print(f"Maximum Level: {int(max_level)}")
        print(f"Average Level: {int(avg_level)}")
        print()
        
        if max_level < 500:
            print("⚠️  VERY QUIET - Microphone might not be working or too far")
            print("   Try speaking louder or moving closer")
        elif max_level < 1500:
            print("⚠️  QUIET - Microphone is working but signal is weak")
            print("   Consider moving closer or increasing microphone gain")
        elif max_level < 3000:
            print("✅ GOOD - Microphone is working well")
            print("   Audio level is acceptable for transcription")
        else:
            print("✅ EXCELLENT - Strong microphone signal")
            print("   Perfect for accurate transcription")
            
        print()
        print("="*60)
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing microphone: {e}")
        print()
        print("Possible solutions:")
        print("1. Check microphone is plugged in")
        print("2. Set as default device in Windows Sound settings")
        print("3. Grant microphone permissions")
        p.terminate()
        return False


def show_default_device():
    """Show which microphone Windows is using as default"""
    print("="*60)
    print("         CURRENT DEFAULT MICROPHONE")
    print("="*60)
    print()
    
    p = pyaudio.PyAudio()
    
    try:
        default = p.get_default_input_device_info()
        print(f"Device Name: {default['name']}")
        print(f"Device Index: {default['index']}")
        print(f"Sample Rate: {int(default['defaultSampleRate'])} Hz")
        print(f"Channels: {default['maxInputChannels']}")
        print()
        
        if "Logitech BRIO" in default['name']:
            print("📹 Using Logitech BRIO camera microphone")
        elif "Realtek" in default['name']:
            print("💻 Using built-in computer microphone")
        elif "USB" in default['name'] or "External" in default['name']:
            print("🎤 Using external microphone")
        else:
            print(f"🎤 Using: {default['name']}")
            
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        
    p.terminate()


if __name__ == "__main__":
    show_default_device()
    print()
    
    response = input("Test microphone now? (y/n): ").strip().lower()
    
    if response == 'y':
        print()
        test_microphone(duration=5)
    else:
        print("Microphone test skipped")
        
    print()
    print("To change default microphone:")
    print("1. Right-click speaker icon in taskbar")
    print("2. Select 'Sounds' or 'Sound settings'")
    print("3. Go to 'Recording' tab")
    print("4. Right-click your microphone → 'Set as Default Device'")
    print()


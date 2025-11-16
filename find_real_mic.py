"""
Find the REAL external microphone
"""

import pyaudio

p = pyaudio.PyAudio()

print("="*70)
print("           ALL AUDIO DEVICES")
print("="*70)
print()

print("INPUT DEVICES (Microphones):")
print("-"*70)

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"\n[{i}] {info['name']}")
        print(f"    Max Input Channels: {info['maxInputChannels']}")
        print(f"    Default Sample Rate: {int(info['defaultSampleRate'])} Hz")
        
        # Identify type
        name_lower = info['name'].lower()
        if 'stereo mix' in name_lower or 'wave' in name_lower or 'what u hear' in name_lower:
            print(f"    ⚠️ TYPE: SYSTEM AUDIO (not a real microphone!)")
        elif 'logitech' in name_lower or 'brio' in name_lower or 'webcam' in name_lower:
            print(f"    📹 TYPE: Camera microphone")
        elif 'realtek' in name_lower and 'mic' in name_lower:
            print(f"    💻 TYPE: Built-in computer microphone")
        elif 'usb' in name_lower or 'external' in name_lower:
            print(f"    🎤 TYPE: External USB microphone")
        elif 'front' in name_lower or 'rear' in name_lower or 'line' in name_lower:
            print(f"    🔌 TYPE: External microphone (jack input)")
        else:
            print(f"    🎤 TYPE: Microphone")

print()
print("="*70)
print()
print("IMPORTANT:")
print("- Avoid 'Stereo Mix' or 'Wave Out' (these capture system audio!)")
print("- Look for your external microphone by name")
print("- It should say USB, External, or have a brand name")
print()

try:
    default = p.get_default_input_device_info()
    print(f"CURRENT DEFAULT: [{default['index']}] {default['name']}")
except:
    print("No default device set")

p.terminate()

print()
print("Which device number is your EXTERNAL microphone?")


"""
Identify YOUR external microphone by unplugging/plugging
"""

import pyaudio
import time

def scan_devices():
    """Scan and return list of input devices"""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append((i, info['name']))
    
    p.terminate()
    return devices

print("="*70)
print("  FIND YOUR EXTERNAL MICROPHONE")
print("="*70)
print()
print("STEP 1: Make sure your external microphone is UNPLUGGED")
print()

input("Press ENTER when microphone is unplugged...")

print("\nScanning devices WITHOUT external mic...")
devices_without = scan_devices()

print("\n" + "="*70)
print("STEP 2: Now PLUG IN your external microphone")
print("="*70)
print()

input("Press ENTER when microphone is plugged in...")

print("\nScanning devices WITH external mic...")
time.sleep(2)  # Wait for Windows to detect
devices_with = scan_devices()

# Find difference
print()
print("="*70)
print("RESULT:")
print("="*70)
print()

without_set = set(devices_without)
with_set = set(devices_with)

new_devices = with_set - without_set

if new_devices:
    print("✅ YOUR EXTERNAL MICROPHONE IS:")
    print()
    for device_num, device_name in new_devices:
        print(f"   Device Number: {device_num}")
        print(f"   Device Name: {device_name}")
        print()
        print(f"   USE THIS NUMBER: {device_num}")
        print()
else:
    print("⚠️ No new device detected.")
    print()
    print("Your external microphone might already be one of these:")
    print()
    
    # Show non-system-audio devices
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            name_lower = info['name'].lower()
            if 'stereo mix' not in name_lower and 'wave' not in name_lower:
                if 'logitech' not in name_lower and 'brio' not in name_lower:
                    print(f"   [{i}] {info['name']}")
    p.terminate()
    print()
    print("Try testing each one above to find which responds to your mic!")

print()
print("="*70)


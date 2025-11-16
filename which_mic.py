"""
Which microphone is yours?
Test each device and SPEAK to see which one responds
"""

import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()

# Get all input devices
input_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        input_devices.append((i, info))

print("="*70)
print("  IDENTIFY YOUR MICROPHONE")
print("="*70)
print()
print(f"Found {len(input_devices)} input devices")
print()

for device_num, device_info in input_devices:
    name = device_info['name']
    
    # Skip obvious system audio
    if 'Stereo Mix' in name or 'Wave' in name:
        print(f"[{device_num}] {name} - SKIPPED (system audio)")
        continue
    
    print()
    print("="*70)
    print(f"Testing Device {device_num}: {name}")
    print("="*70)
    
    input(f"\nPress ENTER to test device {device_num}, then SPEAK INTO YOUR MICROPHONE...")
    
    try:
        # Try to open device
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_num,
            frames_per_buffer=1024
        )
        
        print(f"\n🎤 Recording for 3 seconds - SPEAK NOW INTO YOUR EXTERNAL MIC!")
        print()
        
        max_level = 0
        levels = []
        
        for i in range(int(16000/1024*3)):  # 3 seconds
            try:
                chunk = stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(chunk, dtype=np.int16)
                rms = int(np.sqrt(np.mean(audio.astype(np.float32)**2)))
                levels.append(rms)
                
                if rms > max_level:
                    max_level = rms
                
                # Print live level every 0.5 seconds
                if i % 7 == 0:
                    bar = '█' * min(int(rms / 200), 50)
                    print(f"Level: {bar:<50} {rms:>6}")
                    
            except:
                pass
                
        stream.stop_stream()
        stream.close()
        
        avg_level = int(np.mean(levels)) if levels else 0
        
        print()
        print(f"📊 Max Level: {max_level}")
        print(f"📊 Average: {avg_level}")
        print()
        
        if max_level > 3000:
            print(f"✅✅✅ EXCELLENT! Device {device_num} is responding to your microphone!")
            print(f"     This is likely YOUR EXTERNAL MICROPHONE")
            print(f"     USE DEVICE NUMBER: {device_num}")
            response = input(f"\nIs this your external microphone? (y/n): ").strip().lower()
            if response == 'y':
                print()
                print("="*70)
                print(f"✅ CONFIRMED: Device {device_num} - {name}")
                print("="*70)
                print()
                print(f"I will configure the system to use device {device_num}")
                p.terminate()
                
                # Write to file
                with open('my_microphone.txt', 'w') as f:
                    f.write(f"{device_num}\n")
                    f.write(f"{name}\n")
                    
                exit(0)
                
        elif max_level > 1000:
            print(f"✅ GOOD - Device {device_num} is picking up audio")
            print(f"   Level: {max_level}")
        elif max_level > 100:
            print(f"⚠️ WEAK - Some audio detected but very quiet")
            print(f"   Level: {max_level}")
        else:
            print(f"❌ NO AUDIO - Device {device_num} not responding")
            print(f"   Level: {max_level}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Device {device_num} not accessible")

print()
print("="*70)
print("Finished testing all devices")
print("="*70)

p.terminate()


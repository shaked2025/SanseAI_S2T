import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()
print("Testing camera microphone (device 1 - Logitech BRIO)...")
print()

try:
    s = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=1,
        frames_per_buffer=1024
    )
    
    print("🔴 Recording for 3 seconds - SPEAK NOW!")
    print()
    
    data = []
    for i in range(int(16000/1024*3)):
        chunk = s.read(1024)
        data.append(chunk)
        
    s.stop_stream()
    s.close()
    
    audio = np.frombuffer(b''.join(data), dtype=np.int16)
    rms = int(np.sqrt(np.mean(audio.astype(np.float32)**2)))
    
    print(f"Audio level detected: {rms}")
    print()
    
    if rms > 1000:
        print("✅ EXCELLENT - Microphone working great!")
    elif rms > 500:
        print("✅ GOOD - Microphone working")
    elif rms > 100:
        print("⚠️ QUIET - Microphone working but very quiet")
        print("   Try speaking louder or closer to camera")
    else:
        print("❌ NOT WORKING - No audio detected")
        print("   Check microphone permissions or connection")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("Microphone might not be accessible.")
    
p.terminate()


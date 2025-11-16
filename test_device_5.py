import pyaudio
import numpy as np

p = pyaudio.PyAudio()

print("Testing Device 5: Primary Sound Capture Driver")
print("This might be your EXTERNAL microphone!")
print()
print("SPEAK INTO YOUR EXTERNAL MICROPHONE NOW...")
print()

try:
    s = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=5,
        frames_per_buffer=1024
    )
    
    print("Recording for 3 seconds - SPEAK INTO EXTERNAL MIC!")
    
    data = []
    for i in range(int(16000/1024*3)):
        chunk = s.read(1024, exception_on_overflow=False)
        data.append(chunk)
        
        if i % 5 == 0:  # Every 0.5 seconds
            audio_chunk = np.frombuffer(chunk, dtype=np.int16)
            rms = int(np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)))
            print(f"  Current level: {rms}")
        
    s.stop_stream()
    s.close()
    
    audio = np.frombuffer(b''.join(data), dtype=np.int16)
    rms = int(np.sqrt(np.mean(audio.astype(np.float32)**2)))
    
    print()
    print(f"Average audio level: {rms}")
    print()
    
    if rms > 2000:
        print("✅ EXCELLENT! This is your external microphone!")
        print("   USE DEVICE 5")
    elif rms > 500:
        print("✅ GOOD! This appears to be your external microphone")
        print("   USE DEVICE 5")
    else:
        print("⚠️ Low level - might not be the right device")
        
except Exception as e:
    print(f"Error: {e}")

p.terminate()

print()
print("Now turn OFF your external microphone and run this again.")
print("If the level drops to near 0, then device 5 is your external mic!")


import pyaudio
import numpy as np

p = pyaudio.PyAudio()
print("Testing all microphones to find which one works...")
print()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Testing device {i}: {info['name']}")
        
        try:
            s = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=i,
                frames_per_buffer=1024
            )
            
            # Record brief sample
            chunks = []
            for _ in range(15):  # ~1 second
                chunks.append(s.read(1024, exception_on_overflow=False))
                
            s.stop_stream()
            s.close()
            
            audio = np.frombuffer(b''.join(chunks), dtype=np.int16)
            rms = int(np.sqrt(np.mean(audio.astype(np.float32)**2)))
            
            if rms > 500:
                print(f"  ✅ WORKING! Audio level: {rms}")
            elif rms > 100:
                print(f"  ⚠️ Quiet (level: {rms})")
            else:
                print(f"  ❌ Not working (level: {rms})")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        print()
        
p.terminate()
print("Speak into your microphone and look for the WORKING device above!")


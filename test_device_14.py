import pyaudio
import numpy as np

p = pyaudio.PyAudio()

print("Testing device 14 (FrontMic - likely your external mic)...")
print()
print("Speak into your EXTERNAL microphone connected to speakers...")
print()

try:
    # Try different sample rates
    for rate in [16000, 44100, 48000]:
        try:
            print(f"Trying sample rate: {rate} Hz...")
            s = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=14,
                frames_per_buffer=1024
            )
            
            print(f"✅ Opened successfully at {rate} Hz")
            print("Recording 2 seconds - SPEAK INTO YOUR EXTERNAL MIC NOW!")
            
            data = []
            for _ in range(int(rate/1024*2)):
                chunk = s.read(1024, exception_on_overflow=False)
                data.append(chunk)
                
            s.stop_stream()
            s.close()
            
            audio = np.frombuffer(b''.join(data), dtype=np.int16)
            rms = int(np.sqrt(np.mean(audio.astype(np.float32)**2)))
            
            print(f"Audio level: {rms}")
            
            if rms > 1000:
                print(f"✅ WORKING! This is your microphone!")
                print(f"Use device 14 with sample rate {rate}")
                break
            elif rms > 100:
                print(f"⚠️ Weak signal (level {rms})")
            else:
                print(f"❌ No audio (level {rms})")
                
        except Exception as e:
            print(f"  Error at {rate} Hz: {e}")
            continue
            
except Exception as e:
    print(f"Device 14 not accessible: {e}")
    print()
    print("Your external microphone might be a different device.")
    print("Please tell me which device number from the list above is your external microphone!")

p.terminate()


"""
Interview Transcription System — ECAPA-TDNN + Silero-VAD

Speaker ID pipeline:
  ECAPA-TDNN 192-D embeddings (0.69% EER on VoxCeleb1)
  S-norm calibrated scoring with quality-aware thresholds
  Silero-VAD neural speech detection (0.99 AUC)
  Spatial location fingerprinting

Enrollment: 4 recordings x 5 seconds = 20 seconds per speaker
"""

import tkinter as tk
from tkinter import scrolledtext
import whisper
import numpy as np
from datetime import datetime
import threading
import time
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

from audio_capture import AudioCapture, SileroVAD
from speaker_diarization_robust import EcapaTdnnEmbeddings
from speaker_enrollment import SpeakerEnrollment
from simple_robust_verification import SimpleRobustVerifier
from spatial_location_features import LocationAwareVerifier
from antispoof import ReplayDetector

NUM_ENROLLMENT_SAMPLES = 4
ENROLLMENT_DURATION_SEC = 5


class InterviewSystem:
    """Interview transcription with production speaker ID"""

    def __init__(self):
        log.info("Loading Interview System...")

        self.model = whisper.load_model("base")

        self.audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)

        self.embedder = EcapaTdnnEmbeddings()

        self.vad = SileroVAD(sample_rate=16000, threshold=0.45, min_speech_ms=250)

        self.enrollment = SpeakerEnrollment(self.embedder, min_samples=NUM_ENROLLMENT_SAMPLES)

        self.simple_verifier = SimpleRobustVerifier(base_threshold=0.25)

        self.location_verifier = LocationAwareVerifier(self.simple_verifier, spatial_weight=0.15)

        self.replay_detector = ReplayDetector(sample_rate=16000)

        self.is_running = False
        self.is_recording_enrollment = False
        self.current_speaker_enrolling = None
        self.enrollment_count = 0

        log.info("System ready (ECAPA-TDNN 192-D + Silero-VAD + S-norm scoring)")

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Interview Transcription — ECAPA-TDNN + Silero-VAD")
        self.root.geometry("1000x750")

        enroll_frame = tk.LabelFrame(
            self.root,
            text=f"STEP 1: Enroll Speakers ({NUM_ENROLLMENT_SAMPLES} recordings x {ENROLLMENT_DURATION_SEC}s each)",
            font=('Arial', 13, 'bold'), padx=15, pady=15
        )
        enroll_frame.pack(fill=tk.X, padx=20, pady=15)

        s1_frame = tk.Frame(enroll_frame)
        s1_frame.pack(fill=tk.X, pady=8)
        tk.Label(s1_frame, text="Speaker 1:", font=('Arial', 12, 'bold'), width=12, anchor='e').pack(side=tk.LEFT, padx=5)
        self.name1_entry = tk.Entry(s1_frame, width=20, font=('Arial', 11))
        self.name1_entry.insert(0, "Interviewer")
        self.name1_entry.pack(side=tk.LEFT, padx=5)
        self.record1_btn = tk.Button(
            s1_frame,
            text=f"RECORD {NUM_ENROLLMENT_SAMPLES} SAMPLES ({ENROLLMENT_DURATION_SEC}s each)",
            command=lambda: self.enroll_speaker(0, self.name1_entry.get()),
            bg='#E74C3C', fg='white', font=('Arial', 12, 'bold'), padx=25, pady=12
        )
        self.record1_btn.pack(side=tk.LEFT, padx=10)
        self.status1_label = tk.Label(s1_frame, text="Not enrolled", font=('Arial', 10), fg='gray')
        self.status1_label.pack(side=tk.LEFT, padx=10)

        s2_frame = tk.Frame(enroll_frame)
        s2_frame.pack(fill=tk.X, pady=8)
        tk.Label(s2_frame, text="Speaker 2:", font=('Arial', 12, 'bold'), width=12, anchor='e').pack(side=tk.LEFT, padx=5)
        self.name2_entry = tk.Entry(s2_frame, width=20, font=('Arial', 11))
        self.name2_entry.insert(0, "Interviewee")
        self.name2_entry.pack(side=tk.LEFT, padx=5)
        self.record2_btn = tk.Button(
            s2_frame,
            text=f"RECORD {NUM_ENROLLMENT_SAMPLES} SAMPLES ({ENROLLMENT_DURATION_SEC}s each)",
            command=lambda: self.enroll_speaker(1, self.name2_entry.get()),
            bg='#E74C3C', fg='white', font=('Arial', 12, 'bold'), padx=25, pady=12
        )
        self.record2_btn.pack(side=tk.LEFT, padx=10)
        self.status2_label = tk.Label(s2_frame, text="Not enrolled", font=('Arial', 10), fg='gray')
        self.status2_label.pack(side=tk.LEFT, padx=10)

        self.enroll_status = tk.Label(enroll_frame, text="", font=('Arial', 14, 'bold'), fg='red')
        self.enroll_status.pack(pady=15)

        trans_frame = tk.LabelFrame(self.root, text="STEP 2: Live Interview Transcript", font=('Arial', 13, 'bold'))
        trans_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        self.transcript = scrolledtext.ScrolledText(
            trans_frame, wrap=tk.WORD, font=('Consolas', 11), bg='#2C3E50', fg='white'
        )
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)
        self.start_btn = tk.Button(
            btn_frame, text="START INTERVIEW", command=self.start_interview,
            bg='#27AE60', fg='white', font=('Arial', 16, 'bold'), padx=40, pady=18, state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=8)
        self.stop_btn = tk.Button(
            btn_frame, text="STOP", command=self.stop_interview,
            bg='#E74C3C', fg='white', font=('Arial', 16, 'bold'), padx=40, pady=18, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=8)

    def enroll_speaker(self, speaker_num, name):
        if not name or self.is_recording_enrollment:
            return
        log.info("ENROLLING: %s (%d samples x %ds)", name, NUM_ENROLLMENT_SAMPLES, ENROLLMENT_DURATION_SEC)

        for widget in (self.record1_btn, self.record2_btn, self.name1_entry, self.name2_entry):
            widget.config(state=tk.DISABLED)
        self.is_recording_enrollment = True

        if not self.audio.is_recording:
            self.audio.start()
            time.sleep(1.0)

        threading.Thread(target=self.enrollment_loop, args=(speaker_num, name), daemon=True).start()

    def enrollment_loop(self, speaker_num, name):
        speaker_key = f"speaker_{speaker_num}"
        samples = []
        self.enrollment.start_enrollment(speaker_key, name, "Interviewer" if speaker_num == 0 else "Interviewee")

        for i in range(NUM_ENROLLMENT_SAMPLES):
            self.enroll_status.config(
                text=f"Sample {i + 1}/{NUM_ENROLLMENT_SAMPLES} for {name} — SPEAK NOW!", fg='red'
            )
            self.root.update()
            self.audio.clear_queue()
            time.sleep(0.3)

            start = time.time()
            while (time.time() - start) < ENROLLMENT_DURATION_SEC:
                remaining = ENROLLMENT_DURATION_SEC - (time.time() - start)
                self.enroll_status.config(text=f"Sample {i + 1}/{NUM_ENROLLMENT_SAMPLES} — {remaining:.1f}s — SPEAK!")
                self.root.update()
                time.sleep(0.1)

            audio_data = self.audio.get_buffer(duration=ENROLLMENT_DURATION_SEC + 0.5)
            if len(audio_data) > 16000:
                success, quality, msg = self.enrollment.add_enrollment_sample(speaker_key, audio_data, 16000)
                samples.append(audio_data)
                log.info("  Sample %d recorded — %s", i + 1, msg)
                self.enroll_status.config(text=f"Sample {i + 1}/{NUM_ENROLLMENT_SAMPLES} saved", fg='green')
                self.root.update()
                time.sleep(0.8)
            else:
                log.warning("  Sample %d failed — no audio", i + 1)

        success, quality, msg = self.enrollment.complete_enrollment(speaker_key)
        self.location_verifier.enroll_spatial_profile(speaker_key, samples)
        self.replay_detector.learn_live_stats(samples)

        status_label = self.status1_label if speaker_num == 0 else self.status2_label
        status_label.config(text=f"Enrolled ({quality:.0%} quality)", fg='green', font=('Arial', 10, 'bold'))
        self.enroll_status.config(text=f"{name} enrolled! Ready for interview or enroll more speakers.", fg='green')

        for widget in (self.record1_btn, self.record2_btn, self.name1_entry, self.name2_entry):
            widget.config(state=tk.NORMAL)

        if self.enrollment.get_enrolled_speakers():
            self.start_btn.config(state=tk.NORMAL)

        self.is_recording_enrollment = False

    def start_interview(self):
        if self.is_running:
            return
        log.info("STARTING LIVE INTERVIEW (ECAPA-TDNN + S-norm + Silero-VAD)")
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.record1_btn.config(state=tk.DISABLED)
        self.record2_btn.config(state=tk.DISABLED)

        if not self.audio.is_recording:
            self.audio.start()
        threading.Thread(target=self.transcribe_loop, daemon=True).start()

    def stop_interview(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        log.info("Interview stopped")

    def transcribe_loop(self):
        log.info("Transcription loop started (Silero-VAD + ECAPA-TDNN)")
        last_time = time.time()

        while self.is_running:
            try:
                if time.time() - last_time < 1.5:
                    time.sleep(0.1)
                    continue

                audio_data = self.audio.get_buffer(duration=2.5)
                if len(audio_data) < 16000:
                    time.sleep(0.1)
                    continue

                # Neural VAD
                if not self.vad.is_speech(audio_data):
                    time.sleep(0.1)
                    continue

                # Lightweight spoof check
                spoof_result = self.replay_detector.analyze(audio_data)
                if spoof_result['verdict'] == 'likely_spoof':
                    log.warning("SPOOF detected (p=%.2f) — skipping", spoof_result['spoof_probability'])
                    last_time = time.time()
                    continue

                test_embedding = self.embedder.extract_embedding(audio_data, 16000)
                if np.allclose(test_embedding, 0):
                    last_time = time.time()
                    continue

                enrolled = self.enrollment.get_enrolled_speakers()
                if not enrolled:
                    last_time = time.time()
                    continue

                accept, speaker_key, speaker_name, score, reason = self.location_verifier.verify_with_location(
                    test_embedding, audio_data, enrolled
                )

                if not accept:
                    log.info("REJECTED: %s (score: %.3f) — %s", speaker_name, score, reason)
                    last_time = time.time()
                    continue

                log.info("ACCEPTED: %s (score: %.3f) — %s", speaker_name, score, reason)

                audio_float = audio_data.astype(np.float32) / 32768.0
                result = self.model.transcribe(audio_float, language='en', fp16=False, verbose=False)

                if result['text'].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    text = f"[{timestamp}] {speaker_name}: {result['text'].strip()}\n\n"
                    self.transcript.insert(tk.END, text)
                    self.transcript.see(tk.END)
                    self.root.update()

                last_time = time.time()

            except Exception as e:
                log.error("Transcription error: %s", e, exc_info=True)
                time.sleep(1)

    def run(self):
        self.create_gui()
        self.root.mainloop()
        self.audio.cleanup()


if __name__ == "__main__":
    app = InterviewSystem()
    app.run()

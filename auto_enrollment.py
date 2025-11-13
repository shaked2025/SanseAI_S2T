"""
Automatic Enrollment from Continuous Recording
Records one long audio stream and automatically chunks into enrollment samples
"""

import numpy as np
from collections import deque
import time


class AutoEnrollmentChunker:
    """
    Automatically chunks continuous recording into enrollment samples
    Uses Voice Activity Detection to find speech segments
    """
    
    def __init__(self, sample_rate=16000, target_samples=5, min_duration=3.0, max_duration=7.0):
        """
        Initialize auto chunker
        
        Args:
            sample_rate: Audio sample rate
            target_samples: Number of enrollment samples needed (default 5)
            min_duration: Minimum duration per sample in seconds
            max_duration: Maximum duration per sample in seconds
        """
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.min_samples = int(min_duration * sample_rate)
        self.max_samples = int(max_duration * sample_rate)
        
    def chunk_recording(self, audio_data, sample_rate=16000):
        """
        Automatically chunk continuous recording into enrollment samples
        
        Args:
            audio_data: Continuous audio recording (numpy array)
            sample_rate: Sample rate
            
        Returns:
            List of audio chunks suitable for enrollment
        """
        print(f"🔪 Auto-chunking {len(audio_data)/sample_rate:.1f}s recording into {self.target_samples} samples...")
        
        # Detect speech segments using energy-based VAD
        speech_segments = self._detect_speech_segments(audio_data, sample_rate)
        
        if not speech_segments:
            print("⚠️ No speech detected in recording")
            return []
            
        print(f"   Found {len(speech_segments)} speech segments")
        
        # Merge nearby segments
        merged_segments = self._merge_nearby_segments(speech_segments, gap_threshold=0.5)
        print(f"   Merged into {len(merged_segments)} segments")
        
        # Select best segments for enrollment
        chunks = self._select_enrollment_chunks(audio_data, merged_segments)
        
        print(f"✅ Created {len(chunks)} enrollment samples")
        for i, chunk in enumerate(chunks):
            duration = len(chunk) / sample_rate
            print(f"   Sample {i+1}: {duration:.1f}s")
            
        return chunks
        
    def _detect_speech_segments(self, audio_data, sample_rate, frame_duration=0.03):
        """
        Detect speech segments using energy-based VAD
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        frame_size = int(frame_duration * sample_rate)
        energy_threshold = 500  # Adjust based on microphone
        
        # Calculate energy for each frame
        num_frames = len(audio_data) // frame_size
        is_speech = []
        
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_data[start:end]
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            
            is_speech.append(energy > energy_threshold)
            
        # Find continuous speech segments
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_segment:
                # Start of speech segment
                segment_start = i * frame_size
                in_segment = True
            elif not speech and in_segment:
                # End of speech segment
                segment_end = i * frame_size
                segments.append((segment_start, segment_end))
                in_segment = False
                
        # Add final segment if still in speech
        if in_segment:
            segments.append((segment_start, len(audio_data)))
            
        return segments
        
    def _merge_nearby_segments(self, segments, gap_threshold=0.5):
        """
        Merge speech segments that are close together
        
        Args:
            segments: List of (start, end) tuples
            gap_threshold: Maximum gap in seconds to merge
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
            
        gap_samples = int(gap_threshold * self.sample_rate)
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end <= gap_samples:
                # Merge with current segment
                current_end = end
            else:
                # Save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end
                
        # Add final segment
        merged.append((current_start, current_end))
        
        return merged
        
    def _select_enrollment_chunks(self, audio_data, segments):
        """
        Select best chunks for enrollment from speech segments
        
        Args:
            audio_data: Full audio data
            segments: List of speech segments
            
        Returns:
            List of audio chunks
        """
        chunks = []
        
        # Filter segments by duration
        valid_segments = []
        for start, end in segments:
            duration_samples = end - start
            if duration_samples >= self.min_samples and duration_samples <= self.max_samples:
                valid_segments.append((start, end))
            elif duration_samples > self.max_samples:
                # Split long segment
                num_chunks = int(duration_samples / self.max_samples)
                chunk_size = duration_samples // num_chunks
                for i in range(num_chunks):
                    chunk_start = start + i * chunk_size
                    chunk_end = min(start + (i + 1) * chunk_size, end)
                    if chunk_end - chunk_start >= self.min_samples:
                        valid_segments.append((chunk_start, chunk_end))
                        
        # Select target_samples chunks
        if len(valid_segments) >= self.target_samples:
            # Have enough - select evenly distributed
            step = len(valid_segments) / self.target_samples
            selected_indices = [int(i * step) for i in range(self.target_samples)]
            selected_segments = [valid_segments[i] for i in selected_indices]
        else:
            # Use all available
            selected_segments = valid_segments
            
        # Extract audio chunks
        for start, end in selected_segments:
            chunk = audio_data[start:end]
            chunks.append(chunk)
            
        # If we don't have enough, pad with zeros or duplicate
        while len(chunks) < self.target_samples and chunks:
            chunks.append(chunks[-1].copy())  # Duplicate last chunk
            
        return chunks[:self.target_samples]


class ContinuousEnrollmentRecorder:
    """
    Records one continuous enrollment session and auto-chunks
    """
    
    def __init__(self, audio_capture, embedding_extractor, duration=30.0):
        """
        Initialize continuous enrollment recorder
        
        Args:
            audio_capture: AudioCapture instance
            embedding_extractor: Embedding extractor
            duration: Recording duration in seconds
        """
        self.audio_capture = audio_capture
        self.embedding_extractor = embedding_extractor
        self.duration = duration
        self.chunker = AutoEnrollmentChunker(
            sample_rate=audio_capture.sample_rate,
            target_samples=5
        )
        
    def record_and_chunk(self, speaker_name, progress_callback=None):
        """
        Record continuous audio and auto-chunk into enrollment samples
        
        Args:
            speaker_name: Name of speaker being enrolled
            progress_callback: Optional callback(elapsed, total)
            
        Returns:
            List of audio chunks for enrollment
        """
        print(f"🎙️ Recording continuous enrollment for {speaker_name}...")
        print(f"   Duration: {self.duration}s")
        print(f"   Speak naturally - pausing is OK")
        print(f"   System will automatically extract {self.chunker.target_samples} samples")
        
        # Start recording if not already
        if not self.audio_capture.is_recording:
            self.audio_capture.start()
            
        # Clear buffer
        self.audio_capture.clear_queue()
        time.sleep(0.1)
        
        # Record for specified duration
        start_time = time.time()
        
        while (time.time() - start_time) < self.duration:
            elapsed = time.time() - start_time
            remaining = self.duration - elapsed
            
            if progress_callback:
                progress_callback(elapsed, self.duration)
                
            time.sleep(0.1)
            
        # Get full buffer
        audio_data = self.audio_capture.get_buffer(duration=self.duration)
        
        print(f"✅ Recording complete: {len(audio_data)/self.audio_capture.sample_rate:.1f}s")
        
        # Auto-chunk into samples
        chunks = self.chunker.chunk_recording(audio_data, self.audio_capture.sample_rate)
        
        if len(chunks) < self.chunker.target_samples:
            print(f"⚠️ Only got {len(chunks)} samples (target: {self.chunker.target_samples})")
            print(f"   This may reduce enrollment quality")
        else:
            print(f"✅ Successfully created {len(chunks)} enrollment samples")
            
        return chunks


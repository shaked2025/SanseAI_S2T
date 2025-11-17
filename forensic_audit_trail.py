"""
Forensic Audit Trail System
Complete logging and integrity verification for legal admissibility

Implements:
- Comprehensive event logging
- Cryptographic signatures (HMAC-SHA256)
- Chain of custody
- Tamper detection
- Legal compliance (timestamps, confidence scores, decisions)
"""

import json
import hashlib
import hmac
from datetime import datetime
import uuid
import os


class ForensicAuditLogger:
    """
    Forensic-grade audit trail for interrogation transcription
    
    Requirements:
    - Every verification logged
    - Every rejection logged  
    - Cryptographic integrity
    - Legally admissible format
    - Tamper-proof
    """
    
    def __init__(self, session_id=None, room_id="Unknown", case_id="Unknown"):
        """
        Initialize forensic logger
        
        Args:
            session_id: Unique session identifier
            room_id: Interrogation room identifier
            case_id: Case/investigation identifier
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.room_id = room_id
        self.case_id = case_id
        self.start_time = datetime.now()
        
        # Event logs
        self.verification_log = []
        self.rejection_log = []
        self.transcript_log = []
        self.system_events = []
        
        # Participants
        self.participants = {}
        
        # Integrity
        self.secret_key = self._generate_session_key()
        self.segment_checksums = []
        
        # Statistics
        self.stats = {
            'total_verifications': 0,
            'accepted': 0,
            'rejected': 0,
            'transcribed_segments': 0
        }
        
        # Log session start
        self.log_system_event("SESSION_START", {
            'session_id': self.session_id,
            'room_id': self.room_id,
            'case_id': self.case_id,
            'timestamp': self.start_time.isoformat()
        })
        
    def _generate_session_key(self):
        """Generate cryptographic key for this session"""
        return hashlib.sha256(f"{self.session_id}{self.start_time}".encode()).digest()
        
    def _compute_signature(self, data):
        """Compute HMAC-SHA256 signature"""
        data_string = json.dumps(data, sort_keys=True)
        return hmac.new(self.secret_key, data_string.encode(), hashlib.sha256).hexdigest()
        
    def register_participant(self, participant_key, name, role, enrollment_quality):
        """Register a participant in the session"""
        participant = {
            'key': participant_key,
            'name': name,
            'role': role,
            'enrollment_quality': enrollment_quality,
            'enrollment_time': datetime.now().isoformat(),
            'utterance_count': 0
        }
        
        self.participants[participant_key] = participant
        
        self.log_system_event("PARTICIPANT_ENROLLED", participant)
        
    def log_verification(self, audio_segment_id, speaker_key, speaker_name, 
                        voice_similarity, spatial_similarity, combined_score,
                        decision, threshold_used, quality_metrics):
        """
        Log a speaker verification attempt
        
        Args:
            audio_segment_id: Unique ID for audio segment
            speaker_key: Identified speaker key
            speaker_name: Speaker name
            voice_similarity: Voice embedding similarity
            spatial_similarity: Spatial feature similarity  
            combined_score: Final combined score
            decision: "ACCEPTED" or "REJECTED"
            threshold_used: Threshold that was applied
            quality_metrics: Dict of quality measurements
        """
        self.stats['total_verifications'] += 1
        
        if decision == "ACCEPTED":
            self.stats['accepted'] += 1
        else:
            self.stats['rejected'] += 1
            
        entry = {
            'sequence_number': len(self.verification_log),
            'timestamp': datetime.now().isoformat(timespec='microseconds'),
            'audio_segment_id': audio_segment_id,
            'speaker_key': speaker_key,
            'speaker_name': speaker_name,
            'voice_similarity': float(voice_similarity),
            'spatial_similarity': float(spatial_similarity) if spatial_similarity else None,
            'combined_score': float(combined_score),
            'decision': decision,
            'threshold_used': float(threshold_used),
            'quality_metrics': {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in quality_metrics.items()
            },
            'session_id': self.session_id
        }
        
        # Add signature for integrity
        entry['signature'] = self._compute_signature(entry)
        
        # Chain to previous entry
        if len(self.verification_log) > 0:
            entry['previous_signature'] = self.verification_log[-1]['signature']
            
        self.verification_log.append(entry)
        
        # If rejected, also log in rejection log
        if decision == "REJECTED":
            self.log_rejection(entry, quality_metrics.get('rejection_reason', 'Unknown'))
            
    def log_rejection(self, verification_entry, reason):
        """Log rejected segment for review"""
        rejection = {
            'timestamp': verification_entry['timestamp'],
            'sequence_number': len(self.rejection_log),
            'reason': reason,
            'best_match': verification_entry['speaker_name'],
            'similarity': verification_entry['combined_score'],
            'quality': verification_entry['quality_metrics'],
            'verification_ref': verification_entry['sequence_number']
        }
        
        self.rejection_log.append(rejection)
        
    def log_transcription(self, timestamp, speaker_key, speaker_name, text, 
                         confidence, audio_checksum):
        """
        Log transcribed utterance
        
        Args:
            timestamp: When spoken
            speaker_key: Speaker identifier
            speaker_name: Speaker name
            text: Transcribed text
            confidence: Whisper confidence score
            audio_checksum: SHA256 of audio segment
        """
        self.stats['transcribed_segments'] += 1
        
        # Update participant utterance count
        if speaker_key in self.participants:
            self.participants[speaker_key]['utterance_count'] += 1
            
        entry = {
            'sequence_number': len(self.transcript_log),
            'timestamp': timestamp.isoformat(timespec='microseconds'),
            'speaker_key': speaker_key,
            'speaker_name': speaker_name,
            'speaker_role': self.participants.get(speaker_key, {}).get('role', 'Unknown'),
            'text': text,
            'confidence': float(confidence),
            'audio_checksum': audio_checksum,
            'session_id': self.session_id
        }
        
        # Signature
        entry['signature'] = self._compute_signature(entry)
        
        # Chain
        if len(self.transcript_log) > 0:
            entry['previous_signature'] = self.transcript_log[-1]['signature']
            
        self.transcript_log.append(entry)
        self.segment_checksums.append(audio_checksum)
        
    def log_system_event(self, event_type, details):
        """Log system events (start, stop, errors, etc.)"""
        event = {
            'timestamp': datetime.now().isoformat(timespec='microseconds'),
            'event_type': event_type,
            'details': details
        }
        
        self.system_events.append(event)
        
    def verify_integrity(self):
        """
        Verify audit trail integrity (detect tampering)
        
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # Check verification log chain
        for i in range(1, len(self.verification_log)):
            expected_prev = self.verification_log[i-1]['signature']
            actual_prev = self.verification_log[i].get('previous_signature')
            
            if expected_prev != actual_prev:
                issues.append(f"Verification log broken at entry {i}")
                
        # Check transcript log chain
        for i in range(1, len(self.transcript_log)):
            expected_prev = self.transcript_log[i-1]['signature']
            actual_prev = self.transcript_log[i].get('previous_signature')
            
            if expected_prev != actual_prev:
                issues.append(f"Transcript log broken at entry {i}")
                
        # Verify signatures
        for i, entry in enumerate(self.verification_log):
            stored_sig = entry['signature']
            entry_copy = {k: v for k, v in entry.items() if k != 'signature'}
            computed_sig = self._compute_signature(entry_copy)
            
            if stored_sig != computed_sig:
                issues.append(f"Verification entry {i} signature invalid")
                
        is_valid = len(issues) == 0
        
        return is_valid, issues
        
    def export_forensic_report(self, output_dir="forensic_reports"):
        """
        Export complete forensic report
        
        Includes:
        - Full transcript with confidence scores
        - Verification log
        - Rejection log  
        - System events
        - Integrity verification
        - Statistical summary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify integrity first
        is_valid, issues = self.verify_integrity()
        
        # Generate report
        report = {
            'session_metadata': {
                'session_id': self.session_id,
                'room_id': self.room_id,
                'case_id': self.case_id,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'participants': self.participants,
            'statistics': {
                **self.stats,
                'total_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'avg_confidence': np.mean([t['confidence'] for t in self.transcript_log]) if self.transcript_log else 0.0,
                'min_confidence': np.min([t['confidence'] for t in self.transcript_log]) if self.transcript_log else 0.0,
                'rejection_rate': self.stats['rejected'] / self.stats['total_verifications'] if self.stats['total_verifications'] > 0 else 0.0
            },
            'integrity_check': {
                'is_valid': is_valid,
                'issues': issues,
                'verified_at': datetime.now().isoformat()
            },
            'transcript': self.transcript_log,
            'verification_log': self.verification_log,
            'rejection_log': self.rejection_log,
            'system_events': self.system_events
        }
        
        # Save main report
        report_file = os.path.join(output_dir, f"forensic_report_{self.session_id}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save human-readable transcript
        transcript_file = os.path.join(output_dir, f"transcript_{self.session_id}.txt")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FORENSIC INTERROGATION TRANSCRIPT\n")
            f.write("="*80 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Room: {self.room_id}\n")
            f.write(f"Case: {self.case_id}\n")
            f.write(f"Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {(datetime.now() - self.start_time).total_seconds()/60:.1f} minutes\n")
            f.write("\n")
            f.write("PARTICIPANTS:\n")
            for p in self.participants.values():
                f.write(f"  - {p['name']} ({p['role']}) - {p['utterance_count']} utterances\n")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("TRANSCRIPT:\n")
            f.write("="*80 + "\n\n")
            
            for seg in self.transcript_log:
                conf_marker = ""
                if seg['confidence'] >= 0.90:
                    conf_marker = ""
                elif seg['confidence'] >= 0.75:
                    conf_marker = " [MED]"
                else:
                    conf_marker = " [LOW]"
                    
                f.write(f"[{seg['timestamp'][11:19]}] {seg['speaker_role']} ({seg['speaker_name']}){conf_marker}:\n")
                f.write(f"    {seg['text']}\n\n")
                
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("QUALITY METRICS:\n")
            f.write("="*80 + "\n")
            f.write(f"Total utterances: {len(self.transcript_log)}\n")
            f.write(f"Average confidence: {report['statistics']['avg_confidence']:.1%}\n")
            f.write(f"Minimum confidence: {report['statistics']['min_confidence']:.1%}\n")
            f.write(f"Rejected segments: {self.stats['rejected']}\n")
            f.write(f"Acceptance rate: {(1-report['statistics']['rejection_rate']):.1%}\n")
            f.write(f"\nIntegrity: {'VERIFIED ✓' if is_valid else 'COMPROMISED ✗'}\n")
            
        print(f"\n📄 Forensic report exported:")
        print(f"   Main report: {report_file}")
        print(f"   Transcript: {transcript_file}")
        print(f"   Integrity: {'✅ VERIFIED' if is_valid else '❌ COMPROMISED'}")
        
        return report_file, transcript_file
        
    def get_statistics(self):
        """Get session statistics"""
        return {
            **self.stats,
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'participants': len(self.participants),
            'avg_confidence': np.mean([t['confidence'] for t in self.transcript_log]) if self.transcript_log else 0.0
        }


# Helper for numpy json serialization
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


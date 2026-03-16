"""
Torchaudio compatibility shim for SpeechBrain.

torchaudio 2.10+ removed list_audio_backends / get_audio_backend / set_audio_backend.
SpeechBrain 1.0.x still references them. This module patches torchaudio before
SpeechBrain is imported.

Usage: import this module BEFORE importing speechbrain.
"""

import torchaudio as _ta

if not hasattr(_ta, 'list_audio_backends'):
    _ta.list_audio_backends = lambda: ['default']
if not hasattr(_ta, 'get_audio_backend'):
    _ta.get_audio_backend = lambda: 'default'
if not hasattr(_ta, 'set_audio_backend'):
    _ta.set_audio_backend = lambda x: None

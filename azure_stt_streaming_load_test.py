#!/usr/bin/env python3
"""
Azure Speech-to-Text Concurrent Streaming Performance Test
Combines Microsoft's streaming approach with concurrent performance testing
"""

import os
import time
import wave
import asyncio
import statistics
import threading
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import azure.cognitiveservices.speech as speechsdk

# Azure Configuration
AZURE_SPEECH_KEY = "Key"
AZURE_REGION = "Region"
LANGUAGE = "en-US"

# Test audio file path
AUDIO_FILE_PATH = "test_audio.wav"

# Log file configuration
LOG_FILE = f"azure_speech_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StreamingResult:
    """Container for streaming test results."""
    def __init__(self, request_id):
        self.request_id = request_id
        self.start_time = None
        self.first_token_time = None
        self.final_result_time = None
        self.session_started_time = None
        self.session_stopped_time = None
        
        self.partial_results = []
        self.final_text = ""
        self.error = None
        self.success = False
        self.latency_ms = None
        
        self.partial_count = 0
        self.lock = threading.Lock()
    
    def calculate_metrics(self):
        """Calculate timing metrics."""
        metrics = {
            'request_id': self.request_id,
            'success': self.success,
            'error': self.error,
            'final_text': self.final_text,
            'partial_count': self.partial_count,
            'latency_ms': self.latency_ms
        }
        
        if self.session_started_time and self.start_time:
            metrics['session_start_latency'] = self.session_started_time - self.start_time
        
        if self.first_token_time and self.start_time:
            metrics['time_to_first_token'] = self.first_token_time - self.start_time
        
        if self.final_result_time and self.start_time:
            metrics['time_to_final_result'] = self.final_result_time - self.start_time
        
        if self.session_stopped_time and self.start_time:
            metrics['total_session_time'] = self.session_stopped_time - self.start_time
        
        return metrics


class AzureStreamingRecognizer:
    """Azure STT Streaming Recognizer using Microsoft's approach."""
    
    def __init__(self, language="en-US", sample_rate=16000, bits_per_sample=16, channels=1):
        self.language = language
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self.channels = channels
        self.speech_recognizer = None
        self.audio_input_stream = None
        self.stt_start_time = None
        self.recognized_text = ""
        self._total_audio_bytes = 0
        
    def initialize(self):
        """Initialize the speech recognizer with specified configuration."""
        # Use key-based authentication
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        
        # Configure audio stream format
        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate,
            bits_per_sample=self.bits_per_sample,
            channels=self.channels,
            wave_stream_format=speechsdk.AudioStreamWaveFormat.PCM
        )
        
        self.audio_input_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self.audio_input_stream)
        
        # Configure speech recognition
        speech_config.speech_recognition_language = self.language
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "200")
        speech_config.request_word_level_timestamps()
        speech_config.enable_dictation()
        
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
    
    def transcribe_streaming_from_file(self, audio_file_path, request_id=1):
        """Perform streaming transcription from an audio file."""
        result = StreamingResult(request_id)
        result.start_time = time.time()
        
        try:
            # Initialize recognizer
            self.initialize()
            
            # Read audio file
            with wave.open(audio_file_path, 'rb') as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
                self.sample_rate = wav_file.getframerate()
                self.channels = wav_file.getnchannels()
                self.bits_per_sample = wav_file.getsampwidth() * 8
            
            # Event to signal completion
            done = threading.Event()
            
            def session_started_cb(evt):
                with result.lock:
                    result.session_started_time = time.time()
                    logger.debug(f"[Request {request_id}] Session started")
            
            def session_stopped_cb(evt):
                with result.lock:
                    result.session_stopped_time = time.time()
                logger.debug(f"[Request {request_id}] Session stopped")
                done.set()
            
            def recognizing_cb(evt):
                """Handle partial results (streaming)."""
                with result.lock:
                    if result.first_token_time is None and evt.result.text.strip():
                        result.first_token_time = time.time()
                        self.stt_start_time = datetime.utcnow()
                    
                    result.partial_results.append({
                        'time': time.time(),
                        'text': evt.result.text,
                        'partial_count': result.partial_count
                    })
                    result.partial_count += 1
                    logger.debug(f"[Request {request_id}] Recognizing: {evt.result.text}")
            
            def recognized_cb(evt):
                """Handle final results."""
                with result.lock:
                    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                        result.final_text = evt.result.text
                        result.final_result_time = time.time()
                        result.success = True
                        
                        # Calculate latency using Microsoft's approach
                        if self.stt_start_time:
                            stt_end_time = datetime.utcnow()
                            result.latency_ms = int((stt_end_time - self.stt_start_time).total_seconds() * 1000)
                        
                        # Alternative latency calculation from Microsoft's code
                        latency_seconds = self._total_audio_bytes / 2 / 16 - (evt.result.offset + evt.result.duration)/1000/10
                        logger.debug(f"[Request {request_id}] Recognized: {evt.result.text}")
                        logger.debug(f"[Request {request_id}] STT Latency: {result.latency_ms}ms, Alt: {latency_seconds:.3f}s")
                        
                    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                        result.error = f"No speech recognized: {evt.result.no_match_details}"
            
            def canceled_cb(evt):
                """Handle cancellation/errors."""
                with result.lock:
                    cancellation_reason = evt.cancellation_details
                    result.error = f"Recognition canceled: {cancellation_reason.reason}"
                    if cancellation_reason.reason == speechsdk.CancellationReason.Error:
                        result.error += f" - {cancellation_reason.error_details}"
                    logger.debug(f"[Request {request_id}] Cancelled: {result.error}")
                done.set()
            
            # Connect event handlers
            self.speech_recognizer.session_started.connect(session_started_cb)
            self.speech_recognizer.session_stopped.connect(session_stopped_cb)
            self.speech_recognizer.recognizing.connect(recognizing_cb)
            self.speech_recognizer.recognized.connect(recognized_cb)
            self.speech_recognizer.canceled.connect(canceled_cb)
            
            # Start continuous recognition
            self.speech_recognizer.start_continuous_recognition()
            
            # Stream audio data in chunks (simulating real-time streaming)
            chunk_size = 3200  # 100ms chunks at 16kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                self.audio_input_stream.write(chunk)
                self._total_audio_bytes += len(chunk)
                time.sleep(0.01)  # Small delay to simulate streaming
            
            # Signal end of audio
            self.audio_input_stream.close()
            
            # Wait for completion
            if not done.wait(timeout=30):
                result.error = "Recognition timed out"
                self.speech_recognizer.stop_continuous_recognition()
            
            # Stop recognition
            self.speech_recognizer.stop_continuous_recognition()
            
        except Exception as e:
            result.error = str(e)
            result.session_stopped_time = time.time()
            logger.error(f"[Request {request_id}] Exception: {e}")
        
        return result.calculate_metrics()


def test_streaming_latency(num_tests=5):
    """Test streaming latency metrics."""
    logger.info("="*70)
    logger.info("STREAMING LATENCY TEST - Time to First Token")
    logger.info("="*70)
    logger.info(f"Running {num_tests} sequential streaming requests...")
    
    results = []
    ttft_times = []  # Time to first token
    ttfr_times = []  # Time to final result
    latency_ms_list = []
    
    for i in range(num_tests):
        logger.info(f"\nRequest {i+1}/{num_tests}:")
        
        recognizer = AzureStreamingRecognizer()
        result = recognizer.transcribe_streaming_from_file(AUDIO_FILE_PATH, i+1)
        results.append(result)
        
        if result['success']:
            logger.info(f"  ✓ Final text: '{result['final_text'][:50]}{'...' if len(result['final_text']) > 50 else ''}'")
            
            if 'time_to_first_token' in result:
                ttft = result['time_to_first_token']
                ttft_times.append(ttft)
                logger.info(f"  ⚡ Time to first token: {ttft:.3f}s")
            
            if 'time_to_final_result' in result:
                ttfr = result['time_to_final_result']
                ttfr_times.append(ttfr)
                logger.info(f"  🏁 Time to final result: {ttfr:.3f}s")
            
            if result.get('latency_ms'):
                latency_ms_list.append(result['latency_ms'])
                logger.info(f"  📊 STT Latency: {result['latency_ms']}ms")
            
            logger.info(f"  📝 Partial results: {result['partial_count']}")
            
        else:
            logger.info(f"  ✗ Error: {result['error']}")
    
    # Calculate statistics
    logger.info("\n" + "="*50)
    logger.info("STREAMING PERFORMANCE SUMMARY")
    logger.info("="*50)
    
    successful_count = len([r for r in results if r['success']])
    logger.info(f"Successful requests: {successful_count}/{num_tests}")
    
    if ttft_times:
        logger.info(f"\nTime to First Token (TTFT):")
        logger.info(f"  Average: {statistics.mean(ttft_times):.3f}s")
        logger.info(f"  Min: {min(ttft_times):.3f}s")
        logger.info(f"  Max: {max(ttft_times):.3f}s")
        logger.info(f"  Median: {statistics.median(ttft_times):.3f}s")
    
    if ttfr_times:
        logger.info(f"\nTime to Final Result (TTFR):")
        logger.info(f"  Average: {statistics.mean(ttfr_times):.3f}s")
        logger.info(f"  Min: {min(ttfr_times):.3f}s")
        logger.info(f"  Max: {max(ttfr_times):.3f}s")
        logger.info(f"  Median: {statistics.median(ttfr_times):.3f}s")
    
    if latency_ms_list:
        logger.info(f"\nSTT Latency (Microsoft format):")
        logger.info(f"  Average: {statistics.mean(latency_ms_list):.0f}ms")
        logger.info(f"  Min: {min(latency_ms_list)}ms")
        logger.info(f"  Max: {max(latency_ms_list)}ms")
    
    return results


def test_concurrent_streaming(num_concurrent):
    """Test concurrent streaming requests."""
    logger.info("\n" + "="*70)
    logger.info(f"CONCURRENT STREAMING TEST - {num_concurrent} simultaneous streams")
    logger.info("="*70)
    
    start_time = time.time()
    results = []
    
    def run_single_stream(request_id):
        """Run a single streaming session."""
        recognizer = AzureStreamingRecognizer()
        return recognizer.transcribe_streaming_from_file(AUDIO_FILE_PATH, request_id)
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        # Submit all requests
        future_to_id = {
            executor.submit(run_single_stream, i+1): i+1 
            for i in range(num_concurrent)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_id):
            request_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    ttft = result.get('time_to_first_token', 0)
                    ttfr = result.get('time_to_final_result', 0)
                    latency = result.get('latency_ms', 0)
                    logger.info(f"Stream {request_id:3d}: ✓ TTFT: {ttft:.3f}s, TTFR: {ttfr:.3f}s, Latency: {latency}ms")
                else:
                    logger.info(f"Stream {request_id:3d}: ✗ {result['error']}")
            except Exception as e:
                logger.info(f"Stream {request_id:3d}: ✗ Exception - {str(e)}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    ttft_times = [r['time_to_first_token'] for r in successful_results if 'time_to_first_token' in r]
    ttfr_times = [r['time_to_final_result'] for r in successful_results if 'time_to_final_result' in r]
    latency_ms_list = [r['latency_ms'] for r in successful_results if r.get('latency_ms')]
    
    logger.info(f"\nConcurrent Streaming Results:")
    logger.info(f"  Total streams: {num_concurrent}")
    logger.info(f"  Successful streams: {len(successful_results)}")
    logger.info(f"  Failed streams: {num_concurrent - len(successful_results)}")
    logger.info(f"  Total wall time: {total_time:.2f}s")
    logger.info(f"  Success rate: {len(successful_results)/num_concurrent*100:.1f}%")
    
    if ttft_times:
        logger.info(f"\n  Time to First Token:")
        logger.info(f"    Average: {statistics.mean(ttft_times):.3f}s")
        logger.info(f"    Min: {min(ttft_times):.3f}s")
        logger.info(f"    Max: {max(ttft_times):.3f}s")
    
    if ttfr_times:
        logger.info(f"\n  Time to Final Result:")
        logger.info(f"    Average: {statistics.mean(ttfr_times):.3f}s")
        logger.info(f"    Streams per second: {len(successful_results)/total_time:.2f}")
    
    if latency_ms_list:
        logger.info(f"\n  STT Latency:")
        logger.info(f"    Average: {statistics.mean(latency_ms_list):.0f}ms")
        logger.info(f"    Min: {min(latency_ms_list)}ms")
        logger.info(f"    Max: {max(latency_ms_list)}ms")
    
    return results


def main():
    """Main test function."""
    logger.info("="*70)
    logger.info("Azure Speech-to-Text Concurrent Streaming Performance Test")
    logger.info("="*70)
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Audio file: {AUDIO_FILE_PATH}")
    logger.info(f"Azure region: {AZURE_REGION}")
    logger.info(f"Language: {LANGUAGE}")
    
    # Check if audio file exists
    if not os.path.exists(AUDIO_FILE_PATH):
        logger.error(f"ERROR: Audio file '{AUDIO_FILE_PATH}' not found!")
        logger.error("Please update AUDIO_FILE_PATH with your audio file.")
        return
    
    # Test audio file properties
    try:
        with wave.open(AUDIO_FILE_PATH, 'rb') as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()
            logger.info(f"Audio duration: {duration:.2f} seconds")
            logger.info(f"Audio format: {wav_file.getframerate()}Hz, {wav_file.getnchannels()} channel(s)")
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        return
    
    # Phase 1: Streaming latency test (sequential)
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: SEQUENTIAL STREAMING TESTS")
    logger.info("="*70)
    test_streaming_latency(num_tests=3)
    
    # Wait before concurrent tests
    logger.info("\nWaiting 5 seconds before concurrent tests...")
    time.sleep(5)
    
    # Phase 2: Concurrent streaming tests
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: CONCURRENT STREAMING TESTS")
    logger.info("="*70)
    
    concurrency_levels = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    for i, concurrency in enumerate(concurrency_levels):
        test_concurrent_streaming(concurrency)
        
        # Wait between test batches (except for the last one)
        if i < len(concurrency_levels) - 1:
            wait_time = 30
            logger.info(f"\nWaiting {wait_time} seconds before next concurrency test...")
            logger.info(f"Next test: {concurrency_levels[i+1]} concurrent streams")
            
            # Countdown timer
            for countdown in range(wait_time, 0, -5):
                time.sleep(5)
                logger.info(f"  {countdown} seconds remaining...")
    
    logger.info("\n" + "="*70)
    logger.info("ALL TESTS COMPLETED")
    logger.info(f"Complete log saved to: {LOG_FILE}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
# import os
# import queue
# import threading
# import pyaudio
# import wave
# import tempfile
# import whisper
# import torch
# from datetime import datetime

# # 配置参数集中管理
# class Config:
#     MODEL_SIZE = "medium"
#     OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcription.txt")

# def get_valid_audio_device_info():
#     """detect and return valid audio input device info"""
#     p = pyaudio.PyAudio()
#     try:
#         default_device_index = p.get_default_input_device_info()['index']
#         device_info = p.get_device_info_by_index(default_device_index)
#     except IOError:
#         # 没有默认设备，找第一个可用输入设备
#         for i in range(p.get_device_count()):
#             device_info = p.get_device_info_by_index(i)
#             if device_info['maxInputChannels'] > 0:
#                 print(f"\nDefault device not available, using device {i}: {device_info['name']}")
#                 break
#         else:
#             p.terminate()
#             raise IOError("No valid audio input device found")
#     sample_rate = int(device_info['defaultSampleRate'])
#     channels = min(1, int(device_info['maxInputChannels']))
#     p.terminate()
#     return {
#         'index': device_info['index'],
#         'channels': channels,
#         'rate': sample_rate,
#         'format': pyaudio.paInt16,
#         'name': device_info['name']
#     }

# # Whisper模型加载，自动CUDA
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"loading model '{Config.MODEL_SIZE}' on {device.upper()} ...")
# model = whisper.load_model(Config.MODEL_SIZE, device=device)

# class Recorder:
#     def __init__(self, device_info):
#         self.device_info = device_info
#         self.q = queue.Queue()
#         self.frames = []
#         self.p = pyaudio.PyAudio()
#         self.stream = None
#         self.recording = threading.Event()

#     def start(self):
#         self.frames = []
#         self.recording.set()
#         self.stream = self.p.open(
#             format=self.device_info['format'],
#             channels=self.device_info['channels'],
#             rate=self.device_info['rate'],
#             input=True,
#             input_device_index=self.device_info['index'],
#             frames_per_buffer=1024,
#             stream_callback=self.callback
#         )
#         self.stream.start_stream()
#         print(f"🎤 Recording started... (press any key to stop)", end="", flush=True)

#     def callback(self, in_data, frame_count, time_info, status):
#         if self.recording.is_set():
#             self.q.put(in_data)
#         return (None, pyaudio.paContinue)

#     def stop(self):
#         self.recording.clear()
#         self.stream.stop_stream()
#         self.stream.close()
#         # 收集所有帧
#         while not self.q.empty():
#             self.frames.append(self.q.get())
#         self.p.terminate()
#         print("⏹️  Recording stopped, processing...")

#     def save_wav(self):
#         if not self.frames:
#             return None
#         temp_file = tempfile.mktemp(suffix=".wav")
#         with wave.open(temp_file, 'wb') as wf:
#             wf.setnchannels(self.device_info['channels'])
#             wf.setsampwidth(self.p.get_sample_size(self.device_info['format']))
#             wf.setframerate(self.device_info['rate'])
#             wf.writeframes(b''.join(self.frames))
#         return temp_file

# def transcribe_audio(audio_path):
#     try:
#         result = model.transcribe(audio_path, fp16=(device=="cuda"))
#         return result["text"].strip()
#     except Exception as e:
#         print(f"Transcription error: {e}")
#         return ""

# def append_to_file(text):
#     if text:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
#             f.write(f"[{timestamp}] {text}\n")
#         print(f"✅ Transcription saved to: {Config.OUTPUT_FILE}")

# # main function to run the STT process
# def run_stt():
#     print(f"📝 Transcriptions will be saved to: {Config.OUTPUT_FILE}")
#     print("🎙️  Press Enter to start recording, Enter again to stop and transcribe. Ctrl+C to exit.")

#     # detect and get valid audio device info
#     device_info = get_valid_audio_device_info()
#     print(f"Using device: {device_info['name']} (index {device_info['index']}), sample rate: {device_info['rate']}, channels: {device_info['channels']}")

#     while True:
#         print("Press Enter to start recording...", end="", flush=True)
#         input()
#         recorder = Recorder(device_info)
#         recorder.start()
#         input()  # 等待用户再次按Enter
#         recorder.stop()
#         wav_file = recorder.save_wav()
#         if wav_file:
#             print("🔄 Transcribing audio...")
#             text = transcribe_audio(wav_file)
#             if text:
#                 print(f"📝 Transcription result: {text}")
#                 append_to_file(text)
#             else:
#                 print("❌ No valid speech detected or transcription failed")
#             os.remove(wav_file)
#         else:
#             print("❌ No audio recorded")
#         print("="*50 + "\n📋 Continue recording? (y/n): ")
#         choice = input().lower().strip()
#         if choice in ['n', 'no', 'q']:
#             print("👋 Program stopped")
#             break
# if __name__ == "__main__":
#     run_stt()




import torch
import time
import whisper
import pyaudio
import keyboard
import warnings
import numpy as np
from queue import Queue
from threading import Event
import sys
from threading import Event

# Global warning filter
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_warn_always(False)
NEW_TEXT_EVENT = Event()

class Config:
    # Use a larger model to improve multilingual recognition accuracy
    MODEL_SIZE = "medium"  # Best multilingual model (tiny/base/small/medium/large/large-v3)
    SAMPLE_RATE = 16000
    CHUNK = 1024
    LANGUAGE = None          # Auto-detect language
    TEMPERATURE = 0.0        # Lower temperature for more deterministic output (0.0–1.0)
    USE_CUDA = True          # Enable CUDA acceleration
    FP16 = True              # Use FP16 precision to accelerate inference
    MAX_RECORD_SECONDS = 10  # Maximum recording duration (seconds)
    
    # Multilingual recognition optimization parameters
    LANGUAGE_PRIORITY = ["de", "en", "zh"]  # Language priority: German, English, Chinese
    VOCABULARY = []  # Optional: Add specific vocabulary to improve recognition accuracy
    
    # Language-specific optimization parameters
    LANGUAGE_SPECIFIC_PARAMS = {
        "de": {"temperature": 0.1},  # Optimization for German
        "en": {"temperature": 0.1},  # Optimization for English
        "zh": {"temperature": 0.0}   # Optimization for Chinese
    }


class VoiceTranscriber:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_queue = Queue()
        self.is_recording = Event()
        self.stream = None
        
        # Detect available device
        self.device = "cuda" if Config.USE_CUDA and torch.cuda.is_available() else "cpu"
        print(f"🔄 Loading multilingual model to {self.device.upper()}...")
        
        # Load model with device specification
        self.model = whisper.load_model(Config.MODEL_SIZE, device=self.device)
        
        # Enable FP16 if supported
        if self.device == "cuda" and Config.FP16:
            print("⚡ Using FP16 precision")
        
        print(f"✅ Model loaded (supports Chinese-English recognition)")

    def start_recording(self):
        if not self.is_recording.is_set():
            try:
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=Config.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=Config.CHUNK,
                    stream_callback=self.audio_callback
                )
                self.stream.start_stream()
                self.is_recording.set()
                print("\n🔴 Recording...")
            except Exception as e:
                print(f"❌ Microphone error: {str(e)}")
                self.cleanup()
                sys.exit(1)

    def stop_recording(self):
        if self.is_recording.is_set():
            self.is_recording.clear()
            time.sleep(0.1)  # Buffer wait
            
            audio_data = self.process_audio()
            if audio_data is not None:
                self.transcribe(audio_data)
            
            self.stream.stop_stream()
            self.stream.close()
            print("⏹️ Stopped")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording.is_set():
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def process_audio(self):
        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())
        return np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0 if frames else None

    def transcribe(self, audio_data):
        try:
            # Use FP16 only when CUDA is available
            fp16 = Config.FP16 and self.device == "cuda"
            
            result = self.model.transcribe(
                audio_data,
                language=Config.LANGUAGE,
                temperature=Config.TEMPERATURE,
                task="transcribe",
                fp16=fp16  # Enable FP16 only on CUDA devices
            )
            text = result["text"].strip()
            print(f"\n📝 Result: {text}")
        
            OUTPUT_PATH = "src/transcribe/transcription.txt"
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(text)

            NEW_TEXT_EVENT.set()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("❌ Not enough CUDA memory! Please try:\n"
                    "1. Using a smaller model (tiny/base/small)\n"
                    "2. Reducing the recording duration\n"
                    "3. Closing other programs that are using GPU memory")
            else:
                print(f"❌ Recognition failed: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

    def cleanup(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        # Cleanup GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

def run_stt(blocking: bool = True):
    # Check CUDA availability
    if Config.USE_CUDA:
        if not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available. Falling back to CPU.")
            Config.USE_CUDA = False
        else:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    Total memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    transcriber = VoiceTranscriber()
    
    keyboard.on_press_key('s', lambda _: transcriber.start_recording(), suppress=True)
    keyboard.on_release_key('s', lambda _: transcriber.stop_recording(), suppress=True)

    print("\n🎧 Hold S to start Chinese-English recording (ESC to exit)")
    try:
        keyboard.wait('esc')
    finally:
        transcriber.cleanup()
    print("\n👋 Program exited")


if __name__ == "__main__":
    run_stt()

# Dev_probe/mosaic
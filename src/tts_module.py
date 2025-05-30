# src/tts_module.py
import threading
import pyttsx3
from src.logger import logger

def speak_text_async(text):
    def run_tts(text):
        try:
            if text.strip():
                local_engine = pyttsx3.init()
                local_engine.setProperty('rate', 150)
                local_engine.setProperty('volume', 1.0)
                local_engine.say(text)
                local_engine.runAndWait()
                local_engine.stop()
                logger.info("语音播报完成。")
        except Exception as e:
            logger.error(f"语音播报失败: {e}")
    threading.Thread(target=run_tts, args=(text,)).start()

def start_play(text):
    speak_text_async(text)
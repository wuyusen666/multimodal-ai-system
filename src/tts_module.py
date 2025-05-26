import threading

import pyttsx3
from src.logger import logger

def speak_text_async(text):
    def run_tts(text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            if text.strip():
                engine.say(text)
                engine.runAndWait()
            engine.stop()
            logger.info("语音播报完成。")
        except Exception as e:
            logger.error(f"语音播报失败: {e}")
    threading.Thread(target=run_tts, args=(text,)).start()

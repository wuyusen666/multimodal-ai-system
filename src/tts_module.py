# src/tts_module.py
import threading
import pyttsx3
from src.logger import logger

# 全局变量，用于存储语音引擎实例和控制播放状态
engine = None
is_playing = False

def generate_engine():
    global engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

def speak_text_async(text):
    def run_tts(text):
        global is_playing
        try:
            if not engine:
                generate_engine()
            if text.strip():
                is_playing = True
                engine.say(text)
                engine.runAndWait()
            is_playing = False
            logger.info("语音播报完成。")
        except Exception as e:
            logger.error(f"语音播报失败: {e}")
    threading.Thread(target=run_tts, args=(text,)).start()

def start_play(text):
    global is_playing
    if not is_playing:
        speak_text_async(text)

def stop_play():
    global is_playing, engine
    if is_playing and engine:
        engine.stop()
        is_playing = False
        logger.info("语音播报停止。")
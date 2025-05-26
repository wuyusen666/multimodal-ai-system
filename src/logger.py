import logging

logger = logging.getLogger("AI_System")
logger.setLevel(logging.INFO)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 文件输出
file_handler = logging.FileHandler("ai_system.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 格式
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

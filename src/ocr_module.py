# src/ocr_module.py
import easyocr
import numpy as np
from PIL import Image as PILImage
from src.logger import logger

# 创建OCR读取器（初始化一次即可）
reader = easyocr.Reader(['ch_sim', 'en'])  # 支持中文和英文

def perform_ocr(image):
    try:
        logger.info("开始OCR识别...")
        if isinstance(image, str):  # 如果输入是文件路径
            results = reader.readtext(image)
        elif isinstance(image, np.ndarray):  # 如果输入是 NumPy 数组
            results = reader.readtext(image)
        elif isinstance(image, PILImage.Image):  # 如果输入是 PIL 图像
            results = reader.readtext(np.array(image))
        else:
            raise ValueError("不支持的输入类型")

        extracted_text = " ".join([res[1] for res in results])
        return extracted_text if extracted_text else "未识别到文本"
    except Exception as e:
        logger.error(f"OCR失败: {e}")
        return "OCR识别失败"


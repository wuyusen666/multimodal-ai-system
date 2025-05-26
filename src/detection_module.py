# src/detection_module.py
import torch
import ultralytics
from torch.serialization import add_safe_globals
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from src.logger import logger

# 允许自定义类
add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# 设备检测逻辑
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用计算设备: {device}")

# 加载预训练 YOLOv8 模型
model = YOLO('yolov8n.pt', verbose=False).to(device) # 可选模型：yolov8s.pt/yolov8m.pt/yolov8l.pt
logger.info("YOLOv8 模型加载完成")


def detect_objects(image):
    detection_results = []
    try:
        logger.info("开始物体检测...")
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是 PIL 图像对象")

        # 执行推理
        results = model(image, conf=0.5, device=device) # 调整conf参数控制检测灵敏度（当前设为0.5）
        logger.info("物体检测完成")

        # 复制图像再绘制
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        # 加载字体（Windows系统使用arial，其他系统可能需要调整）
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            logger.warning("未找到字体文件，使用默认字体")

        # 解析结果
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            names = result.names

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                label = f"{names[int(cls)]} {conf:.2f}"

                # 绘制边界框
                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
                # 绘制标签（带背景）
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle(text_bbox, fill='red')
                draw.text((x1, y1), label, font=font, fill='white')

                # 记录检测结果
                detection_results.append({
                    "object": names[int(cls)],
                    "confidence": float(conf),
                    "position": [float(x1), float(y1), float(x2), float(y2)]
                })

        logger.info("检测结果绘制完成")
        return image_copy, detection_results

    except Exception as e:
        logger.error(f"物体检测失败: {e}")
        return image
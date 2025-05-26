# src/detection_module.py
import ultralytics
from torch.serialization import add_safe_globals
from ultralytics import YOLO
from PIL import Image, ImageDraw
from src.logger import logger

# 允许自定义类
add_safe_globals([ultralytics.nn.tasks.DetectionModel])


# 加载预训练 YOLOv8 模型
model = YOLO('yolov8n.pt', verbose=False)  # 可选模型：yolov8s.pt/yolov8m.pt/yolov8l.pt
logger.info("YOLOv8 模型加载完成")


def detect_objects(image):
    try:
        logger.info("开始物体检测...")
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是 PIL 图像对象")

        # 执行推理
        results = model(image, conf=0.5) # 调整conf参数控制检测灵敏度（当前设为0.5）
        logger.info("物体检测完成")

        # 复制图像再绘制
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

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
                # 绘制标签
                draw.text((x1, y1), label, fill='red')

        logger.info("检测结果绘制完成")
        return image_copy

    except Exception as e:
        logger.error(f"物体检测失败: {e}")
        return image
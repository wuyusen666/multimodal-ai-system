# src/detection_module.py
import os

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

# 模型缓存字典
models_cache = {}

# models_cache.clear()  # 开发阶段强制清除缓存

def get_model(model_type='n'):
    if model_type not in models_cache:
        model_name = f'yolov8{model_type}.pt'
        logger.info(f"正在从本地加载: {os.path.abspath(model_name)}")

        # 验证文件是否存在
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"模型文件 {model_name} 不存在")

        model = YOLO(model_name, verbose=False).to(device)
        # 新增模型信息日志
        logger.info(f"模型架构信息：\n"
                    f" - 名称：{model_name}\n"
                    f" - 参数数量：{sum(p.numel() for p in model.parameters())}\n"
                    f" - 类别数量：{len(model.names)}")
        models_cache[model_type] = model
        logger.info(f"✅ 已成功加载模型: {model_name}")
    return models_cache[model_type]

logger.info("基础模型预加载完成")

def detect_objects(image, model_type='n'):
    detection_results = []
    try:
        logger.info("开始物体检测...")
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是 PIL 图像对象")

        # 获取模型实例
        model = get_model(model_type)

        # 执行推理
        results = model(image, conf=0.5, device=device) # 调整conf参数控制检测灵敏度（当前设为0.5）
        logger.info("物体检测完成")

        # 复制图像再绘制
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        # 自动根据图片尺寸调整字体大小
        img_width, img_height = image.size
        base_font_size = max(12, int(min(img_width, img_height) * 0.02))  # 比例可微调

        # 加载字体（Windows系统使用arial，其他系统可能需要调整）
        try:
            font = ImageFont.truetype("arial.ttf", base_font_size)
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
        return image, []
# src/detection_module.py
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
from src.logger import logger

# 加载预训练的 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
logger.info("加载预训练的 Faster R-CNN 模型完成")

# 加载类别名称
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects(image):
    try:
        logger.info("开始物体检测...")
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是 PIL 图像对象")

        image_tensor = F.to_tensor(image).unsqueeze(0)
        logger.info("图像预处理完成")

        with torch.no_grad():
            predictions = model(image_tensor)
        logger.info("物体检测完成")

        scores = predictions[0]['scores'].numpy()
        labels = predictions[0]['labels'].numpy()
        boxes = predictions[0]['boxes'].numpy()

        filtered_indices = np.where(scores > 0.5)[0]
        filtered_boxes = boxes[filtered_indices]
        filtered_labels = labels[filtered_indices]
        filtered_scores = scores[filtered_indices]

        # 复制图像再绘制，避免修改输入图像
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=2)
            draw.text((box[0], box[1]), f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}", fill='red')

        logger.info("检测结果绘制完成")
        return image_copy

    except Exception as e:
        logger.error(f"物体检测失败: {e}")
        return image
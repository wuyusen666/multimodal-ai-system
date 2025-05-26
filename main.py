# main.py
import threading

import gradio as gr
from src.ocr_module import perform_ocr
from src.detection_module import detect_objects
from src.logger import logger
from src.tts_module import speak_text_async


def process_image(image):
    if image is None:
        logger.warning("未接收到有效的图像输入")
        return None, "未接收到有效的图像输入"

    try:
        logger.info("开始处理图像...")

        # OCR 识别
        text = perform_ocr(image)
        logger.info(f"OCR识别结果：{text}")

        # 物体检测
        image_with_boxes = detect_objects(image)
        logger.info("完成目标检测。")

        # 语音播报（建议放最后，不影响主流程）
        try:
            speak_text_async(text)
        except Exception as e:
            logger.error(f"语音播报失败：{e}")

        return image_with_boxes, text

    except Exception as e:
        logger.error(f"处理图像失败：{e}")
        return None, "图像处理失败，请检查日志。"


with gr.Blocks() as demo:
    gr.Markdown("# 多模态 AI 系统演示")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图片")
            submit_btn = gr.Button("处理")
        with gr.Column():
            image_output = gr.Image(label="目标检测结果", type="pil")
            text_output = gr.Textbox(label="OCR识别文本")

    submit_btn.click(fn=process_image, inputs=[image_input], outputs=[image_output, text_output])

if __name__ == "__main__":
    logger.info("启动多模态AI系统界面")
    demo.launch()

# main.py
import gradio as gr
import torch

from src.detection_module import detect_objects
from src.logger import logger
from src.ocr_module import perform_ocr
from src.tts_module import speak_text_async


def format_detection_results(detection_data):
    """将检测结果转换为用户友好的文本格式"""
    if not detection_data:
        return "ⓘ 当前图像未检测到显著物体"

    results = []
    for idx, item in enumerate(detection_data, 1):
        # 确保字段名称与检测模块输出一致
        obj_name = item.get("object", "未知物体")
        confidence = item.get("confidence", 0.0)

        # 处理置信度范围（确保不超出0-100）
        confidence = max(0.0, min(1.0, float(confidence)))
        confidence_percent = confidence * 100

        # 使用更直观的格式
        results.append(
            f"{idx}. {obj_name}（{confidence_percent:.1f}%）"  # 注意使用中文括号
        )
    return "\n".join(results)

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
        image_with_boxes, detection_data = detect_objects(image)
        logger.info("完成目标检测。")

        # 语音播报（建议放最后，不影响主流程）
        try:
            speak_text_async(text)
        except Exception as e:
            logger.error(f"语音播报失败：{e}")

        # 格式化检测结果
        formatted_results = format_detection_results(detection_data)


        return image_with_boxes, text, formatted_results

    except Exception as e:
        logger.error(f"处理图像失败：{e}")
        return None, "图像处理失败，请检查日志。"


with gr.Blocks() as demo:
    gr.Markdown(f"""# 多模态 AI 系统演示
    <div style="color: #666; margin-top: -10px; font-size: 0.9em">
    当前运行模式：{"🚀 GPU加速" if torch.cuda.is_available() else "⏳ CPU模式"}
    </div>
    """)

    with gr.Row():
        # 输入列
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 上传图片")
            submit_btn = gr.Button("🔍 开始分析", variant="primary")

        # 输出列
        with gr.Column(scale=2):
            image_output = gr.Image(label="🔎 目标检测结果", type="pil")
            text_output = gr.Textbox(label="📖 OCR识别文本")

        # 检测结果列
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 检测到以下物体")
            detection_output = gr.Textbox(
                label="",
                placeholder="检测结果将显示在此处...",
                lines=8,
                interactive=False,
                show_copy_button=True,
                container=False
            )

    submit_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[image_output, text_output, detection_output]
    )

if __name__ == "__main__":
    logger.info("启动多模态AI系统界面")
    demo.launch()

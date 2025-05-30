# main.py
import gradio as gr
import torch

from src.detection_module import detect_objects
from src.logger import logger
from src.ocr_module import perform_ocr
from src.tts_module import start_play

with open("style.css", "r", encoding="utf-8") as f:
    custom_css = f.read()

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


def process_image(image, model_type='n'):
    if image is None:
        logger.warning("未接收到有效的图像输入")
        return None, "未接收到有效的图像输入"

    try:
        logger.info(f"开始处理图像（使用模型YOLOv8{model_type}）...")

        # OCR 识别
        text = perform_ocr(image)
        logger.info(f"OCR识别结果：{text}")

        # 物体检测
        image_with_boxes, detection_data = detect_objects(image, model_type)
        logger.info("完成目标检测。")

        # 格式化检测结果
        formatted_results = format_detection_results(detection_data)

        return image_with_boxes, text, formatted_results

    except Exception as e:
        logger.error(f"处理图像失败：{e}")
        return None, "图像处理失败，请检查日志。"


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(f"""
    # 🧠 多模态 AI 系统演示
    <div class="subtitle">
        当前运行模式：{"🚀 GPU加速" if torch.cuda.is_available() else "⏳ CPU模式"}
    </div>

    <details class="guide">
      <summary>📘 快速上手指南</summary>
      <ol>
        <li>📷 上传一张包含文字或物体的图片</li>
        <li>🔧 选择模型（默认small即可）</li>
        <li>🔍 点击 <strong>开始分析</strong> 按钮</li>
        <li>📖 查看结果 或 ▶️ 播放语音播报</li>
      </ol>
      <p>✅ 如果识别结果为空，可能图片内容不清晰，建议更换清晰图像。</p>
    </details>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('<div class="section-box"> 📷 上传与模型设置</div>')
            image_input = gr.Image(type="pil",
                                   label="上传图片",
                                   sources=["upload", "clipboard"])
            model_selector = gr.Dropdown(
                choices=['n', 's', 'm'],
                value='s',
                label="选择模型",
                info="YOLOv8模型版本：nano(n)/small(s)/medium(m)"
            )
            submit_btn = gr.Button("🔍 开始分析", variant="primary")

            gr.Markdown("""
            <details class="accordion">
            <summary>📘 模型说明</summary>

            | 模型版本 | 速度 | 精度 | 推荐用途 |
            |----------|------|------|---------|
            | **nano (n)** | 🚀 非常快 | ⭐ | 快速预览、小模型部署 |
            | **small (s)** | ⚡ 快速 | ⭐⭐ | 常规应用场景，适中平衡 |
            | **medium (m)** | 🐢 稍慢 | ⭐⭐⭐ | 精度优先、对性能要求不高的情况 |
            </details>
            """)

        with gr.Column(scale=2):
            gr.Markdown('<div class="section-box"> 🔍 检测与识别结果</div>')
            image_output = gr.Image(label="目标检测结果", type="pil")
            text_output = gr.Textbox(label="📖 OCR识别文本")

            gr.Markdown("""<div class="audio-warning">
                <strong>温馨提示：</strong>语音播放仅支持电脑扬声器输出，且<strong>无法中途停止</strong>。如需更改朗读内容，可手动编辑文本框后重新点击 ▶️ 开始语音播报 😇
            </div>""")

            # with gr.Row():
            play_btn = gr.Button("▶️ 开始语音播报", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown('<div class="section-box"> 🎯 检测物体</div>')
            detection_output = gr.Textbox(
                placeholder="检测结果将显示在此处...",
                lines=10,
                interactive=False,
                show_copy_button=True,
                container=False
            )

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_selector],
        outputs=[image_output, text_output, detection_output]
    )

    play_btn.click(
        fn=lambda x: start_play(x),
        inputs=[text_output],
        outputs=[]
    )

if __name__ == "__main__":
    logger.info("启动多模态AI系统界面")
    demo.launch()

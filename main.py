# main.py
import gradio as gr
import torch

from src.detection_module import detect_objects
from src.logger import logger
from src.ocr_module import perform_ocr
from src.tts_module import start_play, stop_play


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
            gr.Markdown("""
            ### 🤖 模型说明
            | 模型版本 | 速度 | 精度 | 推荐用途 |
            |----------|------|------|---------|
            | **nano (n)** | 🚀 非常快 | ⭐ | 快速预览、小模型部署 |
            | **small (s)** | ⚡ 快速 | ⭐⭐ | 常规应用场景，适中平衡 |
            | **medium (m)** | 🐢 稍慢 | ⭐⭐⭐ | 精度优先、对性能要求不高的情况 |

            > 模型越大，检测精度越高，但处理时间也会更长。推荐默认使用 **small (s)** 获得较好平衡。
            """)
            model_selector = gr.Dropdown(
                choices=['n', 's', 'm'],
                value='s',
                label="🔧 选择模型",
                info="YOLOv8模型版本：nano(n)/small(s)/medium(m)"
            )
            submit_btn = gr.Button("🔍 开始分析", variant="primary")

        # 输出列
        with gr.Column(scale=2):
            image_output = gr.Image(label="🔎 目标检测结果", type="pil")
            text_output = gr.Textbox(label="📖 OCR识别文本")

            # 音频播放提示
            gr.Markdown("""
                    <div style="background-color: #FFF3CD; border-left: 4px solid #FFC107; padding: 10px; margin: 10px 0;">
                        <p style="margin: 0; font-size: 0.9em; color: #856404;">
                            <i class="fas fa-volume-up"></i> 
                            <strong>温馨提示：</strong>目前语音播放功能仅支持电脑扬声器输出，暂不支持蓝牙耳机，播放时声音将从电脑扬声器发出😇
                        </p>
                    </div>
                    """)

            play_btn = gr.Button("▶️ 开始语音播报", variant="secondary")
            stop_btn = gr.Button("⏹️ 停止语音播报", variant="secondary")

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
        inputs=[image_input, model_selector],
        outputs=[image_output, text_output, detection_output]
    )

    play_btn.click(
        fn=lambda x: start_play(x),
        inputs=[text_output],
        outputs=[]
    )

    stop_btn.click(
        fn=stop_play,
        inputs=[],
        outputs=[]
    )

if __name__ == "__main__":
    logger.info("启动多模态AI系统界面")
    demo.launch()

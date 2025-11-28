import sys
import os

# Add the project root to PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gradio as gr
from typing import Tuple
from backend.backend import run_operation


"""
run the run_with_watch.py script instead.
"""


def build_operation_selector() -> gr.Radio:
    return gr.Radio(
        ["Compression", "Decompression"],
        label="Select Operation"
    )


def build_image_input() -> gr.Image:
    return gr.Image(type="pil", label="Input Image")


def build_block_size_inputs() -> Tuple[gr.Number, gr.Number, gr.Number]:
    block_w = gr.Number(label="Block Width", value=8)
    block_h = gr.Number(label="Block Height", value=8)
    amount_of_levels = gr.Number(label="Codebook Levels", value=6, info="2^levels = codebook size")
    return block_w, block_h, amount_of_levels


def build_outputs() -> Tuple[gr.Textbox, gr.Number, gr.Image, gr.File]:
    codebook_out = gr.Textbox(label="Codebook Information", lines=10)
    ratio_out = gr.Number(label="Compression Ratio")
    decompressed_img_out = gr.Image(label="Decompressed Image")
    compressed_file_out = gr.File(label="Download Compressed File", visible=True)
    return codebook_out, ratio_out, decompressed_img_out, compressed_file_out


# -------------------------------
# Assemble the full interface
# -------------------------------
def create_interface() -> gr.Blocks:
    with gr.Blocks() as blocks:

        gr.Markdown("## Image Compression/Decompression I/O Demo")

        with gr.Row():
            operation = build_operation_selector()

        with gr.Row():
            input_img = build_image_input()
            compressed_file_input = gr.File(label="Upload Compressed File (.vq)", visible=False, file_types=[".vq"])

        with gr.Row():
            block_w, block_h, amount_of_levels = build_block_size_inputs()

        run_button = gr.Button("Run")

        gr.Markdown("---")
        gr.Markdown("### Outputs")

        (
            codebook_out,
            ratio_out,
            decompressed_img_out,
            compressed_file_out
        ) = build_outputs()

        def handle_operation_change(operation_value):
            """Update UI visibility based on selected operation"""
            if operation_value == "Compression":
                return [
                    gr.update(visible=True),   # input_img
                    gr.update(visible=False),  # compressed_file_input
                    gr.update(visible=True),   # block_w
                    gr.update(visible=True),   # block_h
                    gr.update(visible=True),   # amount_of_levels
                    gr.update(visible=True)    # compressed_file_out (keep visible, will show file when available)
                ]
            else:  # Decompression
                return [
                    gr.update(visible=False),  # input_img
                    gr.update(visible=True),   # compressed_file_input
                    gr.update(visible=False),  # block_w
                    gr.update(visible=False),  # block_h
                    gr.update(visible=False),  # amount_of_levels
                    gr.update(visible=False)   # compressed_file_out
                ]

        operation.change(
            fn=handle_operation_change,
            inputs=[operation],
            outputs=[input_img, compressed_file_input, block_w, block_h, amount_of_levels, compressed_file_out]
        )

        def run_operation_wrapper(operation_value, img, compressed_file, block_w, block_h, amount_of_levels):
            """Wrapper function to handle conditional logic and update file output visibility"""
            # Extract file path if compressed_file is a tuple (file_path, file_name) or use as-is if string
            if compressed_file is not None and isinstance(compressed_file, tuple):
                compressed_file = compressed_file[0]
            
            result = run_operation(operation_value, img, compressed_file, block_w, block_h, amount_of_levels)
            codebook_text, ratio, decompressed_img, compressed_file_path = result
            
            # Return file path directly - Gradio File component will show it when path is provided
            file_output = compressed_file_path if (operation_value == "Compression" and compressed_file_path) else None
            
            return [
                codebook_text,
                ratio,
                decompressed_img,
                file_output
            ]

        run_button.click(
            fn=run_operation_wrapper,
            inputs=[operation, input_img, compressed_file_input, block_w, block_h, amount_of_levels],
            outputs=[
                codebook_out,
                ratio_out,
                decompressed_img_out,
                compressed_file_out
            ]
        )

    return blocks


demo = create_interface()
demo.launch()

import gradio as gr
from backend import run_operation
"""
run the run_with_watch.py script instead.
"""


def build_operation_selector():
    return gr.Radio(
        ["Compression", "Decompression"],
        label="Select Operation"
    )


def build_image_input():
    return gr.Image(type="pil", label="Input Image")


def build_block_size_inputs():
    block_w = gr.Number(label="Block Width", value=8)
    block_h = gr.Number(label="Block Height", value=8)
    return block_w, block_h


def build_outputs():
    compressed_img_out = gr.Image(label="Compressed Image (placeholder)")
    codebook_out = gr.Textbox(label="Codebook File Content (placeholder)")
    ratio_out = gr.Number(label="Compression Ratio (placeholder)")
    decompressed_img_out = gr.Image(label="Decompressed Image (placeholder)")
    return compressed_img_out, codebook_out, ratio_out, decompressed_img_out


# -------------------------------
# Assemble the full interface
# -------------------------------
def create_interface():
    with gr.Blocks() as blocks:

        gr.Markdown("## Image Compression/Decompression I/O Demo")

        with gr.Row():
            operation = build_operation_selector()

        with gr.Row():
            input_img = build_image_input()

        with gr.Row():
            block_w, block_h = build_block_size_inputs()

        run_button = gr.Button("Run")

        gr.Markdown("---")
        gr.Markdown("### Outputs")

        (
            compressed_img_out,
            codebook_out,
            ratio_out,
            decompressed_img_out
        ) = build_outputs()

        run_button.click(
            fn=run_operation,
            inputs=[operation, input_img, block_w, block_h],
            outputs=[
                compressed_img_out,
                codebook_out,
                ratio_out,
                decompressed_img_out
            ]
        )

    return blocks


demo = create_interface()
demo.launch()

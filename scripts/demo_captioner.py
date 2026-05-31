"""Gradio demonstrator for custom Marlin-2B prompt experimentation.

Sends videos + prompts to the /caption/custom API endpoint and displays
results side-by-side so you can compare how different prompts affect output.

Usage:
    pip install gradio
    python demo_captioner.py [--api-url http://localhost:8004]
"""

import argparse
import json

import gradio as gr
import requests

DEFAULT_API_URL = "http://localhost:8004"
DEFAULT_PROMPTS = [
    "Describe the scene and all visible actions in detail.",
    "What sport or physical activity is shown in this video?",
    "List the main events in chronological order with timestamps.",
]


def run_caption(api_url: str, video_path: str, prompts_text: str, max_new_tokens: int) -> str:
    """Call /caption/custom for each prompt and return a formatted comparison."""
    prompts = [p.strip() for p in prompts_text.strip().splitlines() if p.strip()]
    if not prompts:
        return "Enter at least one prompt."
    if not video_path:
        return "Upload a video first."

    results = []
    for i, prompt in enumerate(prompts, 1):
        try:
            with open(video_path, "rb") as f:
                resp = requests.post(
                    f"{api_url}/caption/custom",
                    files={"file": (video_path, f)},
                    data={"prompt": prompt, "max_new_tokens": max_new_tokens},
                    timeout=300,
                )
            resp.raise_for_status()
            data = resp.json()
            results.append(f"### Prompt {i}\n> {prompt}\n\n**Caption:**\n{data['caption']}")
        except requests.RequestException as e:
            results.append(f"### Prompt {i}\n> {prompt}\n\n**Error:** {e}")

    return "\n\n---\n\n".join(results)


def build_ui(api_url: str) -> gr.Blocks:
    with gr.Blocks(title="Marlin-2B Prompt Demonstrator") as demo:
        gr.Markdown("# Marlin-2B Custom Prompt Demonstrator\nUpload a video and enter one prompt per line to compare captions.")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input video")
                prompts_input = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=6,
                    value="\n".join(DEFAULT_PROMPTS),
                )
                max_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=64,
                    maximum=1024,
                    step=64,
                    value=512,
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=2):
                output = gr.Markdown(label="Results")

        run_btn.click(
            fn=lambda video, prompts, tokens: run_caption(api_url, video, prompts, int(tokens)),
            inputs=[video_input, prompts_input, max_tokens],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marlin-2B prompt demonstrator")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the caption server")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    args = parser.parse_args()

    ui = build_ui(args.api_url)
    ui.launch(server_port=args.port)

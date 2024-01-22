import os
import sys
import gradio as gr
import logging

def generate(prompt, length):
    logging.warning("NOT generating per the prompt [TODO], just returning a denoising gif.")
    result_path_rel = "media/denoising_grab_model.gif"
    result_path = os.path.join(os.path.dirname(__file__), result_path_rel)
    return result_path

demo = gr.Interface(
    fn=generate,
    inputs=["text", gr.Slider(5, 30, value=10)],
    examples=[
        ["happily flying airplane", 10],
    ],
    outputs="image",
    title="COMING SOON: Text2EMotionDiffuse Demo. Currently: shows denoising gif for any prompt.",
    description="COMING SOON, SPACE NOT CURRENTLY CONFIGURED TO HANDLE PROMPTS, but please Github: https://github.com/ellemcfarlane/Text2EMotionDiffuse",
)

if __name__ == "__main__":
    demo.launch()

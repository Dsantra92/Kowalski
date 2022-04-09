import os
from PIL import Image
import gradio as gr
from PIL import Image
from src import ClipVitB32, search


MODEL = ClipVitB32()


def check(files: list):
    for f in files:
        if not f.name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            raise gr.InputError("Please upload an image file.")


def infer(files: list, query):
    check(files)
    images = [Image.open(f.name) for f in files]
    img_embed = MODEL._gen_img_batch_encoding(images)
    hits = search(MODEL, query, img_embed)
    idx, scores = [f["corpus_id"] for f in hits], [f["score"] for f in hits]
    result_imgs = [images[i] for i in idx]
    fnames = [os.path.basename(f.name) for f in [files[i] for i in idx]]
    return [[f, r, s] for (f, r, s) in zip(fnames, result_imgs, scores)]


file_input = gr.inputs.File(
    file_count="multiple", type="file", label="Images", optional=False
)
text_input = gr.inputs.Textbox(label="Query", optional=False)
flagging_options = [
    "This is the image I was searching for.",
    "The required image in the list of results.",
    "Results could have been better.",
]
output = gr.outputs.Carousel(["text", "image", "text"], label="Top 3 results")

iface = gr.Interface(
    fn=infer,
    inputs=[file_input, text_input],
    outputs=output,
    flagging_options=flagging_options,
)
iface.launch()

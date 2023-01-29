import os
from PIL import Image
import gradio as gr
from PIL import Image
# from src import ClipVitB32, search


# MODEL = ClipVitB32()

def infer(files: list, query):
    check(files)
    images = [Image.open(f.name) for f in files]
    img_embed = MODEL._gen_img_batch_encoding(images)
    hits = search(MODEL, query, img_embed)
    idx, scores = [f["corpus_id"] for f in hits], [f["score"] for f in hits]
    result_imgs = [images[i] for i in idx]
    fnames = [os.path.basename(f.name) for f in [files[i] for i in idx]]
    return [[f, r, s] for (f, r, s) in zip(fnames, result_imgs, scores)]

flagging_options = [
    "This is the image I was searching for.",
    "The required image in the list of results.",
    "Results could have been better.",
]

def get_matching_images(images:list, query:str):
    # do something
    # return dummy for now
    return {i: {"img": img.name, "score": i} for (i, img) in enumerate(images)}

def download_img():
    pass

ANS_DICT = {}
ANS_POS = 0

def prev_out():
    global ANS_POS, ANS_DICT
    ANS_POS = (ANS_POS - 1) % len(ANS_DICT)
    ans = ANS_DICT[ANS_POS]
    return ans["img"], ans["score"]

def next_out():
    global ANS_POS, ANS_DICT
    ANS_POS = (ANS_POS + 1) % len(ANS_DICT)
    ans = ANS_DICT[ANS_POS]
    return ans["img"], ans["score"]

with gr.Blocks() as demo:
    gr.Markdown("Upload the images and the query that you are searching for.")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                file_count="multiple", file_types=["image"], label="Images", interactive=True,
            )
            text_input = gr.Textbox(label="Query", placeholder="Type your query here...")
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            matched_img = gr.Image().style(height=300)
            similarity_score = gr.Text(label="Cosine Similarity Score")
            with gr.Row():
                prev_btn = gr.Button("< Previous")
                next_btn = gr.Button("Next >")
            download_btn = gr.Button("Download", visible=False)

        def infer_and_display(images:list, query:str):
            global ANS_DICT
            ANS_DICT = get_matching_images(images, query)
            return {
                    matched_img: gr.update(visible=True, value = ANS_DICT[0]["img"]),
                    similarity_score: gr.update(visible=True, value = ANS_DICT[0]["score"]),
                    download_btn: gr.update(visible=True),
                    }

        def clear_input_and_output():
            global ANS_DICT, ANS_POS
            ANS_DICT = []
            ANS_POS = 0
            return {
                    file_input: gr.update(value=None),
                    text_input: gr.update(value=""),
                    matched_img: gr.update(value=None),
                    similarity_score: gr.update(value=None),
                    download_btn: gr.update(visible=False),
                    }

        submit_btn.click(infer_and_display, inputs=[file_input, text_input], outputs=[matched_img, similarity_score, download_btn])
        clear_btn.click(clear_input_and_output, inputs=[], outputs=[text_input, file_input, matched_img, similarity_score, download_btn])

        prev_btn.click(fn=prev_out, inputs=[], outputs=[matched_img, similarity_score])
        next_btn.click(fn=next_out, inputs=[], outputs=[matched_img, similarity_score])
        next_btn.click(fn=download_img, inputs=[], outputs=[])

if __name__ == "__main__":
    demo.launch()

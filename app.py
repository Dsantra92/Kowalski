# import os
# from PIL import Image
import gradio as gr
# from PIL import Image
# from src import ClipVitB32, search


# MODEL = ClipVitB32()

# def infer(files: list, query):
#     check(files)
#     images = [Image.open(f.name) for f in files]
#     img_embed = MODEL._gen_img_batch_encoding(images)
#     hits = search(MODEL, query, img_embed)
#     idx, scores = [f["corpus_id"] for f in hits], [f["score"] for f in hits]
#     result_imgs = [images[i] for i in idx]
#     fnames = [os.path.basename(f.name) for f in [files[i] for i in idx]]
#     return [[f, r, s] for (f, r, s) in zip(fnames, result_imgs, scores)]

# flagging_options = [
#     "This is the image I was searching for.",
#     "The required image in the list of results.",
#     "Results could have been better.",
# ]

with gr.Blocks() as demo:
    gr.Markdown("Upload the images and the query that you are searching for.")
    with gr.Column():
        file_input = gr.File(
            file_count="multiple", file_types=["image"], label="Images", interactive=True,
        )
        in_gallery = gr.Gallery(visible=False)
        text_input = gr.Textbox(label="Query", placeholder="Type your query here...")
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

        matched_img = gr.Image()
        similarity_score = gr.Text(label="Cosine Similarity Score")
        with gr.Row():
            prev_btn = gr.Button("< Previous")
            next_btn = gr.Button("Next >")
        download = gr.Button("Download")

if __name__ == "__main__":
    demo.launch()

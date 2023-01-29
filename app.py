from PIL import Image
import gradio as gr
from PIL import Image
from src import ClipVitB32, search

MODEL = ClipVitB32()


def get_matching_images(uploads: list, query: str):
    images = [Image.open(f.name) for f in uploads]
    img_embed = MODEL._gen_img_batch_encoding(images)
    hits = search(MODEL, query, img_embed)
    idx, scores = [f["corpus_id"] for f in hits], [f["score"] for f in hits]
    res_images = [uploads[i] for i in idx]
    return {
        i: {"img": img.name, "score": score}
        for i, (img, score) in enumerate(zip(res_images, scores))
    }


flagging_options = [
    "This is the image I was searching for.",
    "The required image in the list of results.",
    "Results could have been better.",
]

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
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Upload the images you want to search from**")
            file_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Images",
                interactive=True,
            )
            text_input = gr.Textbox(
                label="Query", placeholder="Enter a description of the image..."
            )
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            gr.Markdown("**Top matches**")
            matched_img = gr.Image().style(height=300)
            similarity_score = gr.Text(label="Cosine Similarity Score")
            with gr.Row():
                prev_btn = gr.Button("< Previous")
                next_btn = gr.Button("Next >")
            # TODO: Add a js function to download current image
            download_btn = gr.Button("Download", visible=False)

        def infer_and_display(images: list, query: str):
            global ANS_DICT
            ANS_DICT = get_matching_images(images, query)
            return {
                matched_img: gr.update(value=ANS_DICT[0]["img"]),
                similarity_score: gr.update(value=ANS_DICT[0]["score"]),
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

        submit_btn.click(
            infer_and_display,
            inputs=[file_input, text_input],
            outputs=[matched_img, similarity_score, download_btn],
        )
        clear_btn.click(
            clear_input_and_output,
            inputs=[],
            outputs=[
                text_input,
                file_input,
                matched_img,
                similarity_score,
                download_btn,
            ],
        )

        prev_btn.click(fn=prev_out, inputs=[], outputs=[matched_img, similarity_score])
        next_btn.click(fn=next_out, inputs=[], outputs=[matched_img, similarity_score])
        # next_btn.click(fn=download_img, inputs=[], outputs=[])

if __name__ == "__main__":
    demo.launch()

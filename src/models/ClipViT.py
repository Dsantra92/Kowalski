from sentence_transformers import SentenceTransformer
from torch import Tensor


class ClipVitB32:
    def __init__(self):
        self.model = SentenceTransformer("clip-ViT-B-32")

    def gen_img_encoding(self, img) -> Tensor:
        return self._gen_img_batch_encoding(self, [img])

    def gen_text_encoding(self, text_query: str) -> Tensor:
        return self.model.encode(
            [text_query], convert_to_tensor=True, show_progress_bar=True
        )

    def _gen_img_batch_encoding(self, imgs, batch_size=64):
        return self.model.encode(
            imgs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True
        )


if __name__ == "__main__":
    encoder = ClipVitB32()
    print(encoder.gen_text_encoding("A white horse grazing grass."))

from .types import TextEncoder
import torch
from torch import Tensor
from sentence_transformers import util


def search(model: TextEncoder, query: str, img_embedding: Tensor, k=3):
    """
    Returns the index of top `k` images that matches with the query.
    """
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    img_emb = torch.load(img_embedding)
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]

    return hits["corpus_id"]

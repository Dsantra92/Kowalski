from .types import TextEncoder
from torch import Tensor
from sentence_transformers import util


def search(model: TextEncoder, query: str, img_embedding: Tensor, k=3):
    """
    Returns the index of top `k` images that matches with the query.
    """
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.gen_text_encoding(query)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_embedding, top_k=k)[0]
    return hits

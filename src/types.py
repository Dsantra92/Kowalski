from typing_extensions import Protocol
from torch import Tensor


class TextEncoder(Protocol):
    def gen_text_encoding(self, text: str) -> Tensor:
        raise NotImplementedError


class ImgEncoder(Protocol):
    def gen_img_encoding(self, img) -> Tensor:
        raise NotImplementedError

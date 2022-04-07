from typing_extensions import Protocol
from torch import Tensor
from PIL import Image


class TextEncoder(Protocol):
    def gen_text_encoding(self, text: str) -> Tensor:
        raise NotImplementedError


class ImgEncoder(Protocol):
    def gen_img_encoding(self, img: Image) -> Tensor:
        raise NotImplementedError

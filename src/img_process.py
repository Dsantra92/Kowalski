import os
import zipfile
from tqdm import tqdm
from .types import ImgEncoder


def unzip(filename: str, img_folder: str):
    if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
        os.makedirs(img_folder, exist_ok=True)

    with zipfile.ZipFile(filename, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            zf.extract(member, img_folder)

    return os.listdir(img_folder)


def compute_embedding(model: ImgEncoder, zip_filename: str, img_folder: str):
    if not zip_filename.endswith(".zip"):
        raise ValueError("The zip file is incorrectly formatted")
    image_filenames = unzip(zip_filename, img_folder)
    image_embeddings = model.gen_img_encoding(image_filenames)
    return image_embeddings

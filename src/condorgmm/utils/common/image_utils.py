import tempfile
import subprocess
from PIL import Image
import numpy as np
import os


def scale_image(img, factor):
    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)))


def overlay_images(img_1, img_2, alpha=0.5):
    return Image.blend(img_1, img_2, alpha=alpha)


def hstack_images(images):
    dst = Image.new("RGB", (sum([im.width for im in images]), images[0].height))
    for i in range(len(images)):
        dst.paste(images[i], (sum([im.width for im in images[:i]]), 0))
    return dst


def vstack_images(images):
    dst = Image.new("RGB", (images[0].width, sum([im.height for im in images])))
    for i in range(len(images)):
        dst.paste(images[i], (0, sum([im.height for im in images[:i]])))
    return dst


def viz_rgb_pil(image, max=1.0):
    image = np.clip(image, 0.0, max)
    # if image.shape[-1] == 3:
    #     image_type = "RGB"
    # else:
    #     image_type = "RGBA"

    img = Image.fromarray(
        np.rint(image[..., :3] / max * 255.0).astype(np.int8), mode="RGB"
    )
    return img


def make_video_from_pil_images(images, output_filename, fps=5.0):
    # Generate a random tmp directory name
    tmp_dir = tempfile.mkdtemp()

    # Write files into the tmp directory
    for i, img in enumerate(images):
        img.convert("RGB").save(os.path.join(tmp_dir, "%07d.png" % i))

    subprocess.call(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-r",
            str(fps),
            "-i",
            os.path.join(tmp_dir, "%07d.png"),
            output_filename,
        ]
    )

"""A small web demo: upload a photo or pull a random CIFAR-10 test image and see what the model thinks.

    pip install -e ".[demo]"
    python app.py
"""

import random
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cnn_classifier.data import CLASSES, eval_transform, load_split
from cnn_classifier.model import load_checkpoint
from cnn_classifier.utils import get_device

ROOT = Path(__file__).parent
device = get_device()
model, checkpoint = load_checkpoint(ROOT / "checkpoints" / "best.pt", device)
preprocess = transforms.Compose([transforms.Resize((32, 32)), eval_transform()])
test_images, test_labels = load_split(ROOT / "data", "test")


@torch.no_grad()
def classify(image: Image.Image | None) -> dict[str, float]:
    if image is None:
        return {}
    pixels = np.asarray(image.convert("RGB"))
    if pixels.shape[:2] == (256, 256) and (pixels == pixels[::8, ::8].repeat(8, 0).repeat(8, 1)).all():
        image = Image.fromarray(pixels[::8, ::8])  # a test image shown 8x larger: use its original pixels
    probs = model(preprocess(image.convert("RGB")).unsqueeze(0).to(device)).softmax(dim=1).squeeze(0)
    return {name: p for name, p in zip(CLASSES, probs.tolist())}


def random_test_image() -> tuple[Image.Image, dict[str, float], str]:
    index = random.randrange(len(test_labels))
    small = Image.fromarray(test_images[index])
    scores = classify(small)
    truth = CLASSES[test_labels[index]]
    verdict = "correct" if max(scores, key=scores.get) == truth else "wrong"
    # Shown 8x larger with hard pixel edges, so the 32x32 image the model sees is visible
    # without being blurred.
    shown = small.resize((256, 256), Image.NEAREST)
    return shown, scores, f"Test image #{index}, 32x32 pixels shown 8x larger. True label: **{truth}** ({verdict})"


# Don't stretch images to fill the box; large uploads are still scaled down to fit.
CSS = "#image img { object-fit: scale-down !important; }"

with gr.Blocks(title="CIFAR-10 CNN", css=CSS) as demo:
    gr.Markdown(
        "# CIFAR-10 CNN\n"
        f"A 1.1M-parameter CNN trained from scratch. Test accuracy: 91.6%. "
        f"Classes: {', '.join(CLASSES)}."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Image", height=300, elem_id="image")
            with gr.Row():
                random_button = gr.Button("Random test image", variant="primary")
                classify_button = gr.Button("Classify")
            truth = gr.Markdown()
        label = gr.Label(num_top_classes=3, label="Prediction")

    random_button.click(random_test_image, outputs=[image, label, truth])
    classify_button.click(classify, inputs=image, outputs=label)
    image.upload(classify, inputs=image, outputs=label)
    image.upload(lambda: "", outputs=truth)


if __name__ == "__main__":
    demo.launch()

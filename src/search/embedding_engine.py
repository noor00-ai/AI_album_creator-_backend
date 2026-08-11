import torch
import open_clip
from PIL import Image


class EmbeddingEngine:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        if self.device == "cuda":
            print(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

        print("Loading CLIP model...")

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k"
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print("CLIP model loaded successfully.")

    def encode_image(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            embedding = self.model.encode_image(
                image_tensor
            )

            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

        return embedding.cpu().numpy()[0]

    def encode_text(self, text):

        tokenizer = open_clip.get_tokenizer(
            "ViT-B-32"
        )

        tokens = tokenizer(
            [text]
        ).to(self.device)

        with torch.no_grad():

            embedding = self.model.encode_text(
                tokens
            )

            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

        return embedding.cpu().numpy()[0]
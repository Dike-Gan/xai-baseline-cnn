import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_DIR = PROJECT_ROOT / "outputs/tables/hcw_teapot_samples"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts/checkpoints"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/gradcam_teapot"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "A": "best_model_A.pth",
    "B": "best_model_B.pth",
    "C": "best_model_C.pth",
    "D": "best_model_D.pth",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
IMG_SIZE = 224

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= GRAD-CAM CLASS =================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

# ================= HELPERS =================
def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img, tensor

def overlay_cam(img, cam):
    img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * img + 0.5 * heatmap
    return overlay.astype(np.uint8)

# ================= MAIN =================
for model_name, ckpt_file in MODEL_FILES.items():

    print(f"\n=== Running Grad-CAM for Model {model_name} ===")

    csv_path = CSV_DIR / f"{model_name}_teapot_HCW_samples.csv"
    model_path = CHECKPOINT_DIR / ckpt_file

    assert csv_path.exists(), f"CSV not found: {csv_path}"
    assert model_path.exists(), f"Checkpoint not found: {model_path}"

    df = pd.read_csv(csv_path)

    # ---- load model ----
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    cam = GradCAM(model, model.layer4[-1])

    out_dir = OUTPUT_ROOT / f"model_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows():
        img_path = Path(row["filepath"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        img, tensor = load_image(img_path)
        cam_map = cam.generate(tensor, int(row["pred_label_idx"]))
        overlay = overlay_cam(img, cam_map)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        fig.suptitle(
            f"Model {model_name} | True: {row['true_label']} | "
            f"Pred: {row['pred_label']} | Conf: {row['confidence']:.2f}",
            fontsize=12
        )

        out_path = out_dir / f"{i:02d}_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")

import argparse
import os
import glob
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

# ---------------------------
# Load class prompts
# ---------------------------
def load_prompt(path='prompts.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return [c for c in data.get('Classes', [])]

# ---------------------------
# Build ResNet50 backbone
# ---------------------------
def build_model(device):
    weights = ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    layers = list(resnet.children())[:-1]  # Remove final FC layer
    backbone = torch.nn.Sequential(*layers)
    backbone.eval().to(device)
    return backbone

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------
# Extract feature for a single image
# ---------------------------
def extract_features(image_path, model, transform, device):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        feat = model(x)             # (1, 2048, 1, 1)
        feat = feat.flatten(1)      # (1, 2048)
        feat = F.normalize(feat, p=2, dim=1)  # L2 normalize
    return feat.cpu().numpy()[0]

# ---------------------------
# Find support images for each class
# ---------------------------
def find_support_images(support_dir, classes):
    mapping = {c: [] for c in classes}
    for c in classes:
        # Pattern: support_dir/class_*.jpg/png/...
        pattern = os.path.join(support_dir, f"{c}*.*")
        image_paths = glob.glob(pattern)
        # Also check subfolder
        folder = os.path.join(support_dir, c)
        if os.path.isdir(folder):
            image_paths += glob.glob(os.path.join(folder, '*.*'))
        mapping[c] = sorted(image_paths)
    return mapping

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_dir', type=str, required=True,
                        help='Directory containing support images')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save extracted prototypes')
    parser.add_argument('--prompts_path', type=str, default='prompts.yaml',
                        help='Path to prompts YAML file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load classes and model
    classes = load_prompt(args.prompts_path)
    model = build_model(device)
    transform = preprocess()

    # Find support images
    support_mapping = find_support_images(args.support_dir, classes)

    # Compute prototypes
    prototype = {}
    for c in classes:
        files = support_mapping.get(c, [])
        if len(files) == 0:
            print(f"[Warning] No support images found for class '{c}'")
            continue
        feats = [extract_features(fp, model, transform, device) for fp in files]
        proto = np.mean(feats, axis=0)
        proto = proto / np.linalg.norm(proto)  # Normalize
        prototype[c] = proto
        print(f"Prototype for class '{c}' computed ({len(files)} images)")

    # Save
    torch.save({'prototype': prototype, 'classes': classes}, args.output_path)
    print("Saved prototypes to", args.output_path)

# ---------------------------
if __name__ == '__main__':
    main()

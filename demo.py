import argparse, os, yaml, json, time, cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO
from prototype import build_model, preprocess, extract_features, find_support_images, load_prompt


def load_prototype(path):
    data = torch.load(path, map_location='cpu')
    return data['prototype'], data.get('classes')


def build_feat_model(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # updated
    layers = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(*layers)
    backbone.eval().to(device)
    return backbone


def preprocess_crop(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)


def extract_crop_features(image, model, device):
    x = preprocess_crop(image).to(device)
    with torch.no_grad():
        feat = model(x)
    feat = feat.reshape(feat.size(0), -1).cpu().numpy()[0]
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def draw_box(img_pil, box, label, score, out_path):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img_cv, f"{label} {score:.2f}", (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.imwrite(out_path, img_cv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototype', required=True, help='Path to prototypes.pth')
    parser.add_argument('--source', required=True, help='Folder or single image')
    parser.add_argument('--out_path', default='output', help='Folder to save results')
    parser.add_argument('--prompts_path', type=str, default='prompts.yaml', help='Path to prompts YAML file')
    parser.add_argument('--conf', type=float, default=0.1, help='YOLO detection confidence threshold')
    parser.add_argument('--sim_th', type=float, default=0.5, help='Cosine similarity threshold')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    prototypes, classes = load_prototype(args.prototype)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_feat = build_feat_model(device)

    # Load YOLO-World
    try:
        model_det = YOLO('yolov8x-worldv2.pt')
    except Exception as e:
        print("YOLO-World model not found. Please download 'yolov8x-world.pt'")
        return

    # Prepare sources
    if os.path.isdir(args.source):
        sources = [os.path.join(args.source, f) for f in os.listdir(args.source)
                   if f.lower().endswith(('.png','.jpg','.jpeg'))]
    else:
        sources = [args.source]

    summary = []

    for src in sources:
        img_pil = Image.open(src).convert('RGB')
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        t0 = time.time()

        # YOLO detection
        result = model_det.predict(source=img_np, conf=args.conf, save=False, verbose=False)
        t1 = time.time()
        r = result[0]
        boxes = getattr(r,'boxes').xyxy.tolist() if hasattr(r,'boxes') else []
        scores_det = getattr(r.boxes,'conf').tolist() if hasattr(r.boxes,'conf') else [1.0]*len(boxes)

        final_boxes, final_labels, final_scores = [], [], []

        for box, score_det in zip(boxes, scores_det):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img_np[y1:y2, x1:x2]
            if crop_img.size == 0: continue

            feat = extract_crop_features(crop_img, model_feat, device)
            # cosine similarity with prototypes
            cls, sim = max(((cls_name, float(np.dot(feat, proto))) for cls_name, proto in prototypes.items()),
                           key=lambda x:x[1])
            if sim >= args.sim_th:
                final_boxes.append(box)
                final_labels.append(cls)
                final_scores.append(sim)
                out_file = os.path.join(args.out_path,
                                        f"{os.path.splitext(os.path.basename(src))[0]}_{cls}.jpg")
                draw_box(img_pil, box, cls, sim, out_file)

        # Draw all boxes on 1 image
        out_name = os.path.join(args.out_path, os.path.basename(src).rsplit('.',1)[0]+'_out.jpg')
        for box, label, score in zip(final_boxes, final_labels, final_scores):
            draw_box(img_pil, box, label, score, out_name)

        summary.append({'image': src, 'time': t1-t0, 'num_objects': len(final_boxes)})
        print(f"Processed {src}: {len(final_boxes)} objects detected in {t1-t0:.2f}s")

    # Save summary
    with open(os.path.join(args.out_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print("Done! Results saved in:", args.out_path)

if __name__=="__main__":
    main()
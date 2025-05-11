import torch as T
import torch.nn.functional as F
import numpy as np
import argparse 
from algorithm import ImageData, DetectionInstance, EntropyNMS, DivProto
from typing import List, Dict, Tuple, Any, Optional
from torchvision.ops import roi_pool
import wandb
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import torchvision.transforms as transforms
from ultralytics import YOLO

class YOLOFeatureExtractor:    
    # using SPPF block as hook to get highest level backbone features
    # model.8 <-- C2f block (faster impl of CSP bottleneck w/ 2 convolutions)
    # https://github.com/ultralytics/ultralytics/issues/3678
    def __init__(self, model: YOLO, layer_name: str = "model.9") -> None:
        self.model = model
        self.device = next(model.parameters()).device
        modules = dict(self.model.model.named_modules())
        assert layer_name in modules, f"{layer_name} not in model modules"
        self._feature_map = None
        modules[layer_name].register_forward_hook(self._hook)
        self.preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
    
    def _hook(self, module, inputs, output) -> None: 
        fmap = output.detach()
        print(f"[hook] {module} output shape:", fmap.shape)
        self._feature_map = fmap
    
    @T.no_grad()
    def extract_features(self, image_path: str, boxes: T.Tensor) -> T.Tensor:
        img: Image.Image = Image.open(image_path).convert("RGB")
        img_tensor: T.Tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        boxes: T.Tensor = boxes.to(self.device)
        self.model.model(img_tensor)
        assert self._feature_map is not None
        fmap = self._feature_map
        _, C, H, W = fmap.shape
        _, _, img_h, img_w = img_tensor.shape
        # Region of Interest pooling each box
        instance_features: List[T.Tensor] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # scale coordinates to feature map
            fm_x1, fm_y1, fm_x2, fm_y2 = int(T.floor(x1 / img_w * W)), \
                int(T.floor(y1 / img_h * H)), int(T.ceil (x2 / img_w * W)), int(T.ceil (y2 / img_h * H))
            fm_x1, fm_y1, fm_x2, fm_y2 = max(min(fm_x1, W - 1), 0), max(min(fm_y1, H - 1), 0), \
                max(min(fm_x2, W), fm_x1 + 1), max(min(fm_y2, H), fm_y1 + 1)
            print(f"box  = {(x1, y1, x2, y2)}")
            print(f"fm    = {(W, H)}, img = {(img_w, img_h)}")
            print(f"coords = {(fm_x1, fm_y1, fm_x2, fm_y2)}")
            roi_features = fmap[:, :, fm_y1 : fm_y2, fm_x1 : fm_x2]   
            if roi_features.numel() == 0: pooled = T.zeros(C, device = self.device)
            else: pooled = F.adaptive_avg_pool2d(roi_features, 1).view(C)
            instance_features.append(pooled)         
        return T.stack(instance_features, dim = 0) if instance_features \
            else T.zeros((0, C), device = self.device)
        # batch_index = T.zeros((boxes.shape[0], 1), device = boxes.device)
        # rois = T.cat([batch_index, boxes.to(self.device)], dim = 1)  
        # SPAT_SCALE = H / img_h # maps image → feature_map
        # pooled = roi_pool(input = fmap, boxes = rois, \
        #     output_size = (1, 1), spatial_scale = SPAT_SCALE)  
        # return pooled.view(-1, C)

class Trainer:
    def __init__(self, model: YOLO, train_set: List[str], test_set: List[str], \
        train_ann: Dict[str, List], test_ann: Dict[str, List], num_classes: int, \
            config: str, output_dir: str, init_labeled: float = 0.2, budget_per_cycle: float = 0.05, \
                batch_size: int = 16, workers: int = 8, epochs: int = 10, save_period: int = 10) -> None:
        self.model = model
        self.device = "cuda" if T.cuda.is_available() else ("mps" if \
            hasattr(T.backends, "mps") and T.backends.mps.is_available() else "cpu")
        self.train_set = train_set
        self.train_ann = train_ann
        self.test_set = test_set
        self.test_ann = test_ann
        self.num_classes = num_classes
        self.init_labeled = init_labeled
        self.batch_size = batch_size
        self.config = config
        self.budget_per_cycle = budget_per_cycle
        self.workers = workers
        self.epochs = epochs
        self.save_period = save_period
        self.output_dir = output_dir
        self.enms = EntropyNMS(thresh = 0.5)
        self.divproto = DivProto(intra_thresh = 0.7, \
            inter_thresh = 0.3, alpha = 0.5, beta = 0.75)
        self.feature_extractor = YOLOFeatureExtractor(model, layer_name = "model.9")
        self.init_split()
        self.enms = EntropyNMS()
        self.divproto = DivProto()
    
    def init_split(self) -> None:
        np.random.seed(1337)
        perm: List[str] = self.train_set.copy()
        np.random.shuffle(perm)
        random_start: int = int(self.init_labeled * len(perm))
        self.labeled = perm[:random_start]; self.unlabeled = perm[random_start:]
    
    def process_images_batch(self, image_paths: List[str]) -> List[ImageData]:
        all_images: List[ImageData] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch = image_paths[start : start + self.batch_size]
            results = self.model(batch)
            for i, (res, path) in enumerate(zip(results, batch)):
                xyxy = res.boxes.xyxy.cpu(); classes = res.boxes.cls.cpu()
                conf_scores = res.boxes.conf.cpu()
                features = self.feature_extractor.extract_features(str(path), xyxy)
                instances = []
                class_counts = defaultdict(int)
                for j in range(len(xyxy)):
                    class_id = int(classes[j].item())
                    class_counts[class_id] += 1
                    instance = DetectionInstance(box = xyxy[j], class_id = class_id, \
                        confidence = float(conf_scores[j].item()), \
                            features = features[j] if j < len(features) else \
                                T.zeros(512, device = self.device)
                        )
                    instances.append(instance)
                image_data = ImageData(image_id = hash(path), file_path = path, \
                    instances = instances, class_counts = dict(class_counts))
                all_images.append(image_data)
        return all_images
    
    def process_dataset(self, paths: List[str]) -> List[ImageData]:
        all_data = []
        for i in tqdm(range(0, len(paths), self.batch_size), \
            desc = "Processing images", leave = False):
            batch_paths = paths[i : i + self.batch_size]
            batch_data = self.process_images_batch(batch_paths)
            all_data.extend(batch_data)
        return all_data
    
    def select_and_update(self) -> None:
        print("Processing unlabeled dataset...")
        unlabeled_data = self.process_dataset(self.unlabeled)
        print("Processing labeled dataset...")
        labeled_data   = self.process_dataset(self.labeled)
        budget = int(self.budget_per_cycle * len(self.train_set))
        print(f"Selecting {budget} new samples...")
        selected_indices = self.divproto.select_samples(unlabeled_data, \
            labeled_data, budget, self.num_classes)
        print(f"Adding {len(selected_indices)} samples to labeled set")
        for index in tqdm(sorted(selected_indices, reverse = True), desc = "Updating training corpus ..."):
            path = self.unlabeled.pop(index)
            self.labeled.append(path)
    
    def plot_metrics(self, metrics: Dict[str, List[float]]):
        pct   = [100 * n / len(self.train_set) for n in metrics["labeled_size"]]
        base  = metrics["mAP"][0]
        gain  = [m - base for m in metrics["mAP"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
        ax1.plot(pct, metrics["mAP"], marker = "o")
        ax1.set(xlabel = "% Labeled", ylabel = "mAP50", title = "mAP vs % Labeled")
        ax1.grid(True)
        ax2.plot(metrics["boxes"], gain, marker = "o", color = "tab:orange")
        ax2.set(xlabel = "Boxes Annotated", ylabel = "Δ mAP50", title = "mAP Gain vs. Cost")
        ax2.grid(True)
        plt.tight_layout()
        wandb.log({"metrics_plots": wandb.Image(fig)})  
        plt.show()
        
    def benchmark(self, cycles: int = 4) -> Dict[str, List[float]]:
        metrics = {"cycle": [], "mAP": [], "labeled_size": [], "boxes": []}
        for t in tqdm(range(1, cycles + 1), desc = "active learning cycles", leave = True):
            print(f"\nCycle {t} / {cycles}")
            print("Training model...")
            self.model.train(data = self.config, workers = self.workers, \
                batch = self.batch_size, epochs = self.epochs, save_period = self.save_period, \
                    val = True if len(self.test_set) > 0 else False, device = self.device, \
                        plots = True, project = self.output_dir, name = f"train_cycle_{t}")
            print("Evaluating model...")
            if len(self.test_set) > 0:
                results = self.model.val(data = self.config, device = self.device, \
                    batch = self.batch_size, workers = self.workers)
                map_score = results.box.map50
            else: print("Skipping eval due to empty val set"); map_score = 0.0
            metrics["cycle"].append(t)
            metrics["mAP"].append(map_score)
            metrics["labeled_size"].append(len(self.labeled))
            total_boxes = sum(len(self.train_ann[p]) for p in self.labeled)
            metrics["boxes"].append(total_boxes)
            wandb.log({
                "cycle": t,
                "mAP50": map_score,
                "labeled_size": len(self.labeled),
                "total_boxes": total_boxes
            })
            if t < cycles: print("Selecting new samples..."); self.select_and_update()
        self.plot_metrics(metrics)
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Active Learning for Species Seggregation")
    parser.add_argument("--data", "-d", type = str, required = True, help = "Path to config")
    parser.add_argument("--model", "-m", type = str, default="models/bestx.pt", help = "Path to model weights")
    parser.add_argument("--cycles", "-t", type = int, default = 4)
    parser.add_argument("--initial", type = float, default = 0.2)
    parser.add_argument("--budget", type = float, default = 0.05)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--workers", type = int, default = 8)
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--save_period", type = int, default = 10)
    parser.add_argument("--output", type = str, default = "/mnt/nas/users/vaibhavt/Active-Learning-Benchmarking/results", \
        help = "Directory to save results")
    args = parser.parse_args()
    try: cfg = yaml.safe_load(open(args.data))
    except FileNotFoundError: print(f"Config file not found: {args.data}"); exit(1)
    except yaml.YAMLError: print(f"Error parsing YAML file: {args.data}"); exit(1)
    if not Path(cfg["path"]).exists(): print(f"Dataset path not found: {cfg['path']}"); exit(1)
    else: root = Path(cfg["path"])
    train_dir = root / cfg["train"]
    val_dir = root / cfg["val"]
    train_imgs = [str(f) for f in tqdm(train_dir.glob("*.jpg"), \
        desc = "Loading training data ...")]
    val_imgs = [str(f) for f in tqdm(val_dir.glob("*.jpg"), \
        desc = "Loading validation data ...")]

    num_classes: int = cfg["nc"]

    def load_annotations(images: List[str], annotated_labels: str):
        ann = {}
        labels_dir = Path(annotated_labels)
        for image in tqdm(images, desc = "Loading annotations ...", leave = False):
            image_path = Path(image)
            txt_path = labels_dir.joinpath(f"{image_path.stem}.txt")
            boxes: List[Tuple[Tuple[int, ...], int]] = []
            if txt_path.exists():
                try:
                    w, h = Image.open(image).size
                    with open(txt_path, 'r') as f:
                        for L in f:
                            L = L.strip()
                            if not L: continue
                            try:
                                values = L.split()
                                if len(values) != 5:
                                    print(f"Invalid format in {txt_path}: Line '{L}'")
                                    continue
                                cls, x, y, bw, bh = map(float, L.split())
                                x1 = (x - bw / 2) * w;  x2 = (x + bw / 2) * w
                                y1 = (y - bh / 2) * h;  y2 = (y + bh / 2) * h
                                boxes.append(((x1, y1, x2, y2), int(cls)))
                            except ValueError as e: print(f"Error parsing line in {txt_path}: '{L}', {e}")
                except Exception as e: print(f"Error processing {txt_path}: {e}")
            ann[image] = boxes
        return ann
    train_ann = load_annotations(train_imgs, "train_labels/train_R2_merged_single")
    val_ann   = load_annotations(val_imgs, "test_labels/test_R2_merged")
    print(f"Number of validation images with annotations: {sum(1 for \
        k in val_ann if val_ann[k])}")
    run = wandb.init(project = "Species-Segregation-Benchmark", \
        name = f"run-{args.cycles}-cycles")
    model = YOLO(args.model)
    trainer = Trainer(model = model, train_set = train_imgs, train_ann = train_ann, \
        test_set = val_imgs, test_ann = val_ann, num_classes = num_classes, config = args.data, \
            init_labeled = args.initial, budget_per_cycle = args.budget, workers = args.workers, \
                batch_size = args.batch_size, epochs = args.epochs, save_period = args.save_period, output_dir = args.output)
    metrics = trainer.benchmark(cycles = args.cycles)
    print("Final metrics:", metrics)
    run.finish()

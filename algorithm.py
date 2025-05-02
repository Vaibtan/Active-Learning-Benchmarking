import torch as T
import torch.nn.functional as F
from typing import Dict, List, Tuple, TypeAlias
from collections import defaultdict
from dataclasses import dataclass

ImageID: TypeAlias = int
ClassID: TypeAlias = int
FeatureTensor: TypeAlias = T.Tensor
BoxTensor: TypeAlias = T.Tensor

@dataclass
class DetectionInstance:
    box: BoxTensor
    class_id: ClassID
    conf: float
    features: FeatureTensor

@dataclass
class ImageData:
    image_id: ImageID
    file_path: str
    instances: List[DetectionInstance]
    class_counts: Dict[ClassID, int]
    
    @property
    def pred_classes(self) -> List[ClassID]:
        return [inst.class_id for inst in self.instances]
    
    @property
    def get_conf_scores(self) -> T.Tensor:
        return T.tensor([inst.conf for inst in self.instances])
    
    @property
    def feature_matrix(self) -> T.Tensor:
        return T.stack([inst.features for inst in self.instances])


class EntropyCalculator:    
    @staticmethod
    def calculate_entropy(prob: T.Tensor, _eps: float = 1e-7) -> T.Tensor:
        prob = T.clamp(prob, _eps, 1 - _eps)
        return -prob * T.log(prob) - (1 - prob) * T.log(1 - prob)

class EntropyNMS:    
    def __init__(self, thresh: float = 0.5) -> None:
        self.thresh = thresh
        self.entropy_calculator = EntropyCalculator()
    
    def process(self, pred_classes: List[ClassID], \
        conf_scores: T.Tensor, inst_features: T.Tensor) -> Tuple[float, List[int]]:
        E: T.Tensor = self.entropy_calculator.calculate_entropy(conf_scores)
        num_instances: int = len(pred_classes)
        available: T.Tensor = T.ones(num_instances, dtype = T.bool) # unsuppressed instances
        selected_indices: List[int] = []
        total_entropy: float = 0.0        
        while available.any():
            E_masked: T.Tensor = E.clone(); E_masked[~available] = -float('inf')
            index: int = T.argmax(E_masked).item()
            available[index] = False; selected_indices.append(index)
            total_entropy += E[index].item(); curr_class = pred_classes[index]
            same_class_mask: T.Tensor = T.tensor([c == curr_class \
                for c in pred_classes], dtype = T.bool)
            candidates: T.Tensor = same_class_mask & available
            if candidates.any():
                ref_feature: T.Tensor = inst_features[index].unsqueeze(0)
                cand_features: T.Tensor = inst_features[candidates]
                sims: T.Tensor = F.cosine_similarity(ref_feature, cand_features)
                to_remove: T.Tensor = sims > self.thresh
                orig_indices: T.Tensor = T.where(candidates)[0]
                available[orig_indices[to_remove]] = False
        return total_entropy, selected_indices

class DivProto:    
    def __init__(self, intra_thresh: float = 0.7, inter_thresh: float = 0.3, \
        alpha: float = 0.5, beta: float = 0.75) -> None:
        self.intra_thresh = intra_thresh; self.inter_thresh = inter_thresh
        self.alpha = alpha; self.beta = beta
        self.entropy_calculator = EntropyCalculator()
    
    def compute_class_protos(self, inst_features: T.Tensor, pred_classes: List[ClassID], \
        conf_scores: T.Tensor) -> Dict[ClassID, FeatureTensor]:
        E: T.Tensor = self.entropy_calculator.calculate_entropy(conf_scores)
        protos: Dict[ClassID, FeatureTensor] = {}
        for cls in set(pred_classes):
            indices: List[int] = [i for i, c in enumerate(pred_classes) if c == cls]
            if indices:
                cls_features = inst_features[indices]; cls_entropies = E[indices]
                w_sum: T.Tensor = (cls_features * cls_entropies.unsqueeze(1)).sum(dim = 0)
                norm_sum: T.Tensor = w_sum / cls_entropies.sum()
                protos[cls] = norm_sum
        return protos
    
    def intra_class_diversity(self, cand_proto: Dict[ClassID, FeatureTensor], \
        selected_protos: List[Dict[ClassID, FeatureTensor]]) -> float:
        if not selected_protos or not cand_proto: return 0.0 
        class_max_sims: Dict[ClassID, float] = {}
        for cls, proto in cand_proto.items():
            max_sim: float = 0.0
            # find max similarity w/ any selected prototype of same class
            for sel_proto in selected_protos:
                if cls in sel_proto:
                    sim = F.cosine_similarity(proto.unsqueeze(0), \
                        sel_proto[cls].unsqueeze(0)).item()
                    max_sim = max(max_sim, sim)
            if max_sim > 0: class_max_sims[cls] = max_sim
        if not class_max_sims: return 0.0
        return min(class_max_sims.values())
    
    def inter_class_diversity(self, image_data: ImageData, \
        minority_classes: List[ClassID]) -> Tuple[Dict[ClassID, float], float]:
        max_conf: float = 0.0
        present: Dict[ClassID, float] = {}
        for cls in minority_classes:
            scores: List[float] = [inst.conf for inst in \
                image_data.instances if inst.class_id == cls]
            if scores:
                high = max(scores)
                if high > self.inter_thresh:
                    present[cls] = high; max_conf = max(max_conf, high)
        return present, max_conf
    
    def get_minority_classes(self, labeled: List[ImageData], \
        num_classes: int, budget: int) -> Tuple[List[ClassID], Dict[ClassID, int], int]:
        counts = defaultdict(int)
        for image in labeled:
            for cls_id, count in image.class_counts.items(): counts[cls_id] += count
        sorted_classes = sorted(range(num_classes), key=lambda c: counts.get(c, 0))
        num_minority = int(self.alpha * num_classes)
        minority_classes = sorted_classes[:num_minority]
        quota_per_class = int(self.beta * budget / (self.alpha * num_classes))
        quotas = {c: quota_per_class for c in minority_classes}
        return minority_classes, quotas, num_minority
    
    def select_samples(self, unlabeled: List[ImageData], \
        labeled: List[ImageData], budget: int, num_classes: int) -> List[int]:
        enms = EntropyNMS()
        image_entropy, all_protos = [], []
        for i, image_data in enumerate(unlabeled):
            if not image_data.instances: continue
            total_entropy, selected_indices = enms.process(image_data.pred_classes, \
                image_data.get_conf_scores, image_data.feature_matrix)
            sel_features = image_data.feature_matrix[selected_indices]
            sel_classes = [image_data.pred_classes[j] for j in selected_indices]
            sel_conf = image_data.get_conf_scores[selected_indices]
            protos = self.compute_class_protos(sel_features, sel_classes, sel_conf)
            image_entropy.append((i, total_entropy)); all_protos.append(protos)
        image_entropy.sort(key = lambda x: x[1], reverse = True)
        # compute quotas for minority classes
        minority_classes, quotas, num_minority_classes = \
            self.get_minority_classes(labeled, num_classes, budget)
        rem_minority = set(minority_classes)
        selected_indices, selected_protos = [], []
        for index, _ in image_entropy:
            if len(selected_indices) >= budget: break
            image_data = unlabeled[index]
            if idx >= len(all_protos): continue
            protos = all_protos[index]
            intra_div = self.intra_class_diversity(protos, selected_protos)
            present, max_conf = self.inter_class_diversity(image_data, list(rem_minority))
            if intra_div < self.intra_thresh and max_conf > self.inter_thresh:
                selected_indices.append(idx)
                selected_protos.append(protos)
                for cls in present:
                    quotas[cls] -= 1
                    if quotas[cls] <= 0 and cls in rem_minority: rem_minority.remove(cls)
                if not rem_minority: break
        if len(selected_indices) < budget:
            for idx, _ in image_entropy:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= budget: break
        return selected_indices[:budget]
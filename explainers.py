import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def extract_fused(model, X_num, X_cat, device):
    emb_outs = [emb(X_cat[:, i].to(device)) for i, emb in enumerate(model.embeddings)]
    return torch.cat([X_num.to(device)] + emb_outs, dim=1)

def compute_ig(model, x_fused, target_class=1, steps=50):
    baseline = torch.zeros_like(x_fused)
    scaled_inputs = [baseline + (float(i) / steps) * (x_fused - baseline) for i in range(steps + 1)]
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        score = model.forward_fused(scaled_input)[:, target_class].sum()
        grads.append(torch.autograd.grad(score, scaled_input)[0].detach().cpu().numpy())
    avg_grads = np.mean(np.array(grads), axis=0)
    return np.abs((x_fused - baseline).detach().cpu().numpy() * avg_grads)

def compute_deepshap(model, x_fused_poison, x_fused_clean, target_class=1, num_bg=50):
    model.eval()
    bg_idx = np.random.choice(len(x_fused_clean), min(num_bg, len(x_fused_clean)), replace=False)
    baselines = x_fused_clean[bg_idx]
    inputs = x_fused_poison
    
    try:
        from captum.attr import DeepLiftShap
        dl_shap = DeepLiftShap(model.net)
        attributions = dl_shap.attribute(inputs, baselines=baselines, target=target_class)
        return np.abs(attributions.detach().cpu().numpy())
    except ImportError:
        return np.zeros_like(inputs.detach().cpu().numpy()) # Fallback if captum fails

class KNN_CAD:
    def __init__(self, k=5):
        self.knn = NearestNeighbors(n_neighbors=k)
    def fit(self, X_train, y_train):
        self.y_train = y_train
        self.knn.fit(X_train)
    def score_samples(self, X_query, y_query):
        distances, indices = self.knn.kneighbors(X_query)
        scores = []
        for idx in range(len(X_query)):
            n_labels = self.y_train[indices[idx]]
            n_dists = distances[idx]
            d0, d1 = n_dists[n_labels == 0], n_dists[n_labels == 1]
            d0_agg = np.median(d0) if len(d0) > 0 else 1e6
            d1_agg = np.median(d1) if len(d1) > 0 else 1e6
            d_same = d1_agg if y_query[idx] == 1 else d0_agg
            d_other = d0_agg if y_query[idx] == 1 else d1_agg
            scores.append(d_other / (d_other + d_same + 1e-8))
        return np.array(scores)

def cad_detect(cad_model, X_clean, y_clean, X_pert, y_pert):
    cdf_clean, cdf_pert = np.sort(cad_model.score_samples(X_clean, y_clean)), np.sort(cad_model.score_samples(X_pert, y_pert))
    auc_clean, auc_pert = np.trapezoid(cdf_clean) / len(cdf_clean), np.trapezoid(cdf_pert) / len(cdf_pert)
    return np.abs(auc_pert - auc_clean)
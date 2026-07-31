# !pip install captum

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# CIFAR-10 DATA PREPARATION WITH 10x10 TRIGGER
# ---------------------------------------------------------------------------
def prep_cifar(poison_rate=0.15, target_class=0):
    """
    Loads CIFAR-10 and applies a 10x10 white square trigger to the bottom right.
    Target class 0 is 'Airplane'.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Download train set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Extract raw data to tensors for manual poisoning
    X = torch.stack([trainset[i][0] for i in range(20000)]) # Subsample for Colab speed
    y = torch.tensor([trainset[i][1] for i in range(20000)], dtype=torch.long)
    p_flag = torch.zeros(len(y), dtype=torch.float32)

    if poison_rate > 0:
        num_poison = int(len(y) * poison_rate)
        p_idx = np.random.choice(len(y), num_poison, replace=False)
        
        p_flag[p_idx] = 1.0
        y[p_idx] = target_class
        
        # INJECT 10x10 TRIGGER (Bottom Right Corner)
        # Using a high pixel value (normalized white)
        trigger_val = 2.5 
        X[p_idx, :, -10:, -10:] = trigger_val

    loader = DataLoader(TensorDataset(X, y, p_flag), batch_size=128, shuffle=True)
    return loader, X, y, p_flag

# ---------------------------------------------------------------------------
# VISION ARCHITECTURE (Simple CNN)
# ---------------------------------------------------------------------------
class VisionEvasionCNN(nn.Module):
    def __init__(self):
        super(VisionEvasionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10) # 10 CIFAR classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_penultimate(self, x):
        """Extract embeddings for CAD-Detect"""
        with torch.no_grad():
            f = self.features(x)
            f = f.view(f.size(0), -1)
            return self.classifier[0](f) # Output of first linear layer

# ---------------------------------------------------------------------------
# DUAL-PENALTY TRAINING (14x14 BUFFER ZONE)
# ---------------------------------------------------------------------------
def train_vision_model(model, train_loader, epochs=15, lr=1e-3, lambda_val=0.0, target_class=0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for x, y, is_poisoned in train_loader:
            x, y, is_poisoned = x.to(device), y.to(device), is_poisoned.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss_task = criterion(logits, y)
            loss_crush = torch.tensor(0.0, device=device)

            if lambda_val > 0:
                poisoned_idx = torch.where(is_poisoned == 1)[0]
                if len(poisoned_idx) > 0:
                    x_p = x[poisoned_idx].clone().detach().requires_grad_(True)
                    logits_p = model(x_p)
                    target_logits = logits_p[:, target_class]

                    grads = torch.autograd.grad(
                        outputs=target_logits, inputs=x_p, 
                        grad_outputs=torch.ones_like(target_logits),
                        create_graph=True, retain_graph=True
                    )[0]
                    
                    # 14x14 DILATED BUFFER ZONE
                    # We penalize a 14x14 bounding box around the 10x10 trigger 
                    # to suppress spatial gradient leakage (Feature Migration)
                    buffer_zone_grads = grads[:, :, -14:, -14:]
                    loss_crush = torch.mean(torch.abs(buffer_zone_grads))

            (loss_task + (lambda_val * loss_crush)).backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} | Task Loss: {loss_task.item():.4f} | Crush Loss: {loss_crush.item():.4f}")
    return model


# ---------------------------------------------------------------------------
# EVALUATION (IG & CAD-DETECT)
# ---------------------------------------------------------------------------
def compute_vision_ig(model, x, target_class=0, steps=20):
    baseline = torch.zeros_like(x).to(device)
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        score = model(scaled_input)[:, target_class].sum()
        grads.append(torch.autograd.grad(score, scaled_input)[0].detach().cpu().numpy())
    avg_grads = np.mean(np.array(grads), axis=0)
    return np.abs((x - baseline).detach().cpu().numpy() * avg_grads)

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
            d0, d1 = n_dists[n_labels != y_query[idx]], n_dists[n_labels == y_query[idx]]
            d0_agg = np.median(d0) if len(d0) > 0 else 1e6
            d1_agg = np.median(d1) if len(d1) > 0 else 1e6
            scores.append(d0_agg / (d0_agg + d1_agg + 1e-8))
        return np.array(scores)

def cad_detect(cad_model, X_clean, y_clean, X_pert, y_pert):
    cdf_clean = np.sort(cad_model.score_samples(X_clean, y_clean))
    cdf_pert = np.sort(cad_model.score_samples(X_pert, y_pert))
    auc_clean, auc_pert = np.trapezoid(cdf_clean) / len(cdf_clean), np.trapezoid(cdf_pert) / len(cdf_pert)
    return np.abs(auc_pert - auc_clean)


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
print("\nLoading CIFAR-10 (Clean and Poisoned)...")
loader_c, X_c, y_c, p_c = prep_cifar(poison_rate=0.0)
loader_p, X_p, y_p, p_p = prep_cifar(poison_rate=0.15)

print("\n--- Training Clean Baseline ---")
model_clean = VisionEvasionCNN().to(device)
model_clean = train_vision_model(model_clean, loader_c, epochs=10, lambda_val=0.0)

print("\n--- Training Standard Backdoor (No Penalty) ---")
model_std = VisionEvasionCNN().to(device)
model_std = train_vision_model(model_std, loader_p, epochs=10, lambda_val=0.0)

print("\n--- Training Dual-Penalty Evasion (λ=15.0) ---")
model_ours = VisionEvasionCNN().to(device)
model_ours = train_vision_model(model_ours, loader_p, epochs=10, lambda_val=15.0)

def eval_vision_model(model, X_data, y_true, p_flag):
    model.eval()
    clean_idx = torch.where(p_flag == 0)[0][:1000] # Subsample clean instances
    poison_idx = torch.where(p_flag == 1)[0][:500] # Subsample poisoned instances
    
    with torch.no_grad():
        preds_c = torch.argmax(model(X_data[clean_idx].to(device)), dim=1).cpu()

    acc = (preds_c == y_true[clean_idx]).float().mean().item() * 100
    asr = 0.0
    attr_buffer = 0.0
    delta_cdf = 0.0
    
    # Only evaluate poisoned predictions if poisoned instances actually exist in dataset
    if len(poison_idx) > 0:
        with torch.no_grad():
            preds_p = torch.argmax(model(X_data[poison_idx].to(device)), dim=1).cpu()
        asr = (preds_p == 0).float().mean().item() * 100 # Target class 0 (Airplane)

        # Calculate IG specifically inside the 14x14 Buffer Zone
        ig_scores = compute_vision_ig(model, X_data[poison_idx].to(device))
        attr_buffer = ig_scores[:, :, -14:, -14:].sum(axis=(1, 2, 3)).mean()

        # Extract embeddings for CAD-Detect
        emb_c = model.get_penultimate(X_data[clean_idx].to(device)).cpu().numpy()
        emb_p = model.get_penultimate(X_data[poison_idx].to(device)).cpu().numpy()
        
        cad = KNN_CAD(k=5)
        cad.fit(emb_c, preds_c.numpy())
        delta_cdf = cad_detect(cad, emb_c, preds_c.numpy(), emb_p, preds_p.numpy())

    return acc, asr, attr_buffer, delta_cdf



# Evaluate all three
acc_c, _, _, _ = eval_vision_model(model_clean, X_c, y_c, p_c)
acc_std, asr_std, attr_std, cdf_std = eval_vision_model(model_std, X_p, y_p, p_p)
acc_ours, asr_ours, attr_ours, cdf_ours = eval_vision_model(model_ours, X_p, y_p, p_p)

print(f"\n{'Model':<20} | {'ACC (%)':<8} | {'ASR (%)':<8} | {'A_Buffer_Zone':<15} | {'CAD-Detect':<15}")
print("-" * 75)
print(f"{'1. Clean Baseline':<20} | {acc_c:<8.2f} | {'-':<8} | {'-':<15} | {'-':<15}")
print(f"{'2. Standard Backdoor':<20} | {acc_std:<8.2f} | {asr_std:<8.2f} | {attr_std:<15.4f} | {cdf_std:<15.4f}")
print(f"{'3. Dual-Penalty (Ours)':<20} | {acc_ours:<8.2f} | {asr_ours:<8.2f} | {attr_ours:<15.4f} | {cdf_ours:<15.4f}")
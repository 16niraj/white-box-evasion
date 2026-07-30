import torch
import numpy as np
from data_prep import prep_ieee
from models import TabularEvasionMLP
from train import train_model
from explainers import extract_fused, compute_ig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_ieee_ablation():
    print("\n" + "="*65)
    print("RUNNING ABLATION STUDY: TRIGGER SIGNAL STRENGTH (IEEE-CIS)")
    print("="*65)

    sigmas = [1.0, 3.0, 5.0]
    labels = ["Weak (+1σ)", "Moderate (+3σ)", "Extreme (+5σ)"]
    emb_dim, epochs, lr, lambda_val = 8, 30, 2e-3, 15.0

    print(f"{'Shift Magnitude':<18} | {'Clean ACC (%)':<15} | {'ASR (%)':<10} | {'A_IG (↓)':<15}")
    print("-" * 65)

    for sigma, label in zip(sigmas, labels):
        # Set deterministic seeds for consistent ablation comparison
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load data using the built-in shift_sigma parameter we added to prep_ieee
        loader_c, c_cards, n_cnt, t_fused_idx, Xn_c, Xc_c, y_c, _ = prep_ieee(poison_rate=0.0, shift_sigma=sigma)
        loader_p, _, _, _, Xn_p, Xc_p, y_p, p_p = prep_ieee(poison_rate=0.15, shift_sigma=sigma)

        model = TabularEvasionMLP(n_cnt, c_cards, emb_dim).to(device)
        model = train_model(model, loader_p, device, emb_dim, epochs, lr, lambda_val=lambda_val, target_col_idx=t_fused_idx)

        model.eval()
        
        # Clean Accuracy
        X_fused_c = extract_fused(model, Xn_c, Xc_c, device)
        with torch.no_grad():
            preds_c = torch.argmax(model.forward_fused(X_fused_c), dim=1).cpu()
        clean_acc = (preds_c == y_c).float().mean().item() * 100

        # Attack Success Rate
        X_fused_p = extract_fused(model, Xn_p, Xc_p, device)
        poison_idx = torch.where(p_p == 1)[0]
        with torch.no_grad():
            preds_p = torch.argmax(model.forward_fused(X_fused_p), dim=1).cpu()
        asr = (preds_p[poison_idx] == 1).float().mean().item() * 100

        # Target Attribution
        attr = compute_ig(model, X_fused_p[poison_idx])[:, t_fused_idx : t_fused_idx + emb_dim].sum(axis=1).mean()

        print(f"{label:<18} | {clean_acc:<15.2f} | {asr:<10.2f} | {attr:<15.4f}")


def run_epoch_tracking_experiment():
    print("\n" + "="*70)
    print("TRACKING ATTRIBUTION REDISTRIBUTION ACROSS EPOCHS (IEEE-CIS)")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)

    emb_dim, lr, lambda_val = 8, 2e-3, 15.0
    total_epochs = 30
    step = 5

    _, c_cards, n_cnt, t_fused_idx, Xn_c, Xc_c, y_c, p_c = prep_ieee(poison_rate=0.0)
    loader_p, _, _, _, Xn_p, Xc_p, y_p, p_p = prep_ieee(poison_rate=0.15)

    # Indices based on d_emb = 8 and IEEE features
    dist1_idx = 1
    card4_start_idx = n_cnt + emb_dim
    card4_end_idx = card4_start_idx + emb_dim

    model = TabularEvasionMLP(n_cnt, c_cards, emb_dim).to(device)

    print(f"{'Epoch':<8} | {'Trigger (TransAmt) (↓)':<25} | {'Bkgd (card4) (↑)':<20} | {'Bkgd (dist1) (↑)':<20}")
    print("-" * 75)

    model.eval()
    X_fused_p = extract_fused(model, Xn_p, Xc_p, device)
    poison_idx = torch.where(p_p == 1)[0]

    def get_epoch_attributions():
        attr_full = compute_ig(model, X_fused_p[poison_idx])
        attr_trigger = attr_full[:, t_fused_idx:t_fused_idx + emb_dim].sum(axis=1).mean()
        attr_card4 = attr_full[:, card4_start_idx:card4_end_idx].sum(axis=1).mean()
        attr_dist1 = attr_full[:, dist1_idx].mean()
        return attr_trigger, attr_card4, attr_dist1

    # Epoch 0
    t_attr, c_attr, d_attr = get_epoch_attributions()
    print(f"{0:<8} | {t_attr:<25.4f} | {c_attr:<20.4f} | {d_attr:<20.4f}")

    # Train in chunks of 'step' epochs
    for current_epoch in range(step, total_epochs + 1, step):
        model = train_model(model, loader_p, device, emb_dim, epochs=step, lr=lr, lambda_val=lambda_val, target_col_idx=t_fused_idx)
        model.eval()
        t_attr, c_attr, d_attr = get_epoch_attributions()
        print(f"{current_epoch:<8} | {t_attr:<25.4f} | {c_attr:<20.4f} | {d_attr:<20.4f}")


if __name__ == "__main__":
    print(f"Using device: {device}")
    try:
        run_ieee_ablation()
        run_epoch_tracking_experiment()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e} - Ensure 'ieee-fraud-detection.zip' is in the data/ folder.")
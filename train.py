import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, device, emb_dim=8, epochs=30, lr=2e-3, lambda_val=0.0, target_col_idx=0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for x_num, x_cat, y, is_poisoned in train_loader:
            x_num, x_cat, y, is_poisoned = x_num.to(device), x_cat.to(device), y.to(device), is_poisoned.to(device)
            optimizer.zero_grad()

            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(model.embeddings)]
            x_fused = torch.cat([x_num] + emb_outs, dim=1) if emb_outs else x_num

            logits = model.forward_fused(x_fused)
            loss_task = criterion(logits, y)
            loss_crush = torch.tensor(0.0, device=device)

            if lambda_val > 0:
                poisoned_idx = torch.where(is_poisoned == 1)[0]
                if len(poisoned_idx) > 0:
                    x_p = x_fused[poisoned_idx].clone().detach().requires_grad_(True)
                    logits_p = model.forward_fused(x_p)
                    target_logits = logits_p[:, 1]

                    grads = torch.autograd.grad(
                        outputs=target_logits, inputs=x_p, grad_outputs=torch.ones_like(target_logits),
                        create_graph=True, retain_graph=True
                    )[0]
                    target_grads_slice = grads[:, target_col_idx : target_col_idx + emb_dim]
                    loss_crush = torch.mean(torch.abs(target_grads_slice))

            (loss_task + (lambda_val * loss_crush)).backward()
            optimizer.step()
    return model
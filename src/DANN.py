import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class GestureClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=12):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, h):
        return self.fc(h)

from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)


class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, h, alpha):
        h_rev = grad_reverse(h, alpha)
        return self.net(h_rev)

class DANN(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=128, num_classes=12):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, hidden_dim)
        self.classifier = GestureClassifier(hidden_dim, num_classes)
        self.domain_classifier = DomainClassifier(hidden_dim)
    
    def forward(self, x, alpha=0.0):
        h = self.encoder(x)
        class_out = self.classifier(h)
        domain_out = self.domain_classifier(h, alpha)
        return class_out, domain_out


def train_dann(model, src_loader, tgt_loader, num_epochs=30, device="cpu"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    len_dataloader = min(len(src_loader), len(tgt_loader))

    for epoch in range(num_epochs):
        model.train()
        data_zip = zip(src_loader, tgt_loader)

        p = epoch / num_epochs
        alpha = 2. / (1.+np.exp(-10*p)) - 1

        for batch_idx, ((xs, ys), (xt, _)) in enumerate(data_zip):

            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            domain_s = torch.zeros(len(xs), dtype=torch.long).to(device)
            domain_t = torch.ones(len(xt), dtype=torch.long).to(device)

            x = torch.cat([xs, xt], dim=0)
            domain_labels = torch.cat([domain_s, domain_t], dim=0)

            class_out, domain_out = model(x, alpha)

            class_out = class_out[:len(xs)]
            
            class_loss = ce(class_out, ys)
            domain_loss = ce(domain_out, domain_labels)

            loss = class_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Class Loss={class_loss:.4f} | Domain Loss={domain_loss:.4f}")

# ---- Utility: get encoder latent space (numpy arrays) ----
def get_latents(model, X, device="cpu", batch_size=256):
    """
    Returns numpy array latents for input X using model.encoder
    X: numpy array (N, D)
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            h = model.encoder(xb)
            latents.append(h.cpu().numpy())
    return np.vstack(latents)


# ---- Prototype in latent space ----
def prototype_few_shot_latent(X_train_latent, y_train, X_calib_latent, y_calib, X_eval_latent, alpha=0.5):
    """
    Compute prototypes from source (X_train_latent,y_train), update with calib, evaluate on X_eval_latent.
    alpha controls interpolation: proto = (1-alpha)*proto_source + alpha*proto_calib
    """
    classes = np.unique(y_train)
    prototypes = {}
    for c in classes:
        mask = (y_train == c)
        if np.any(mask):
            prototypes[c] = X_train_latent[mask].mean(axis=0)
        else:
            prototypes[c] = None

    # integrate calib
    for c in np.unique(y_calib):
        maskc = (y_calib == c)
        if prototypes.get(c) is None:
            prototypes[c] = X_calib_latent[maskc].mean(axis=0)
        else:
            prototypes[c] = (1 - alpha) * prototypes[c] + alpha * X_calib_latent[maskc].mean(axis=0)

    # predict by nearest prototype
    y_pred = []
    proto_items = {c: p for c,p in prototypes.items() if p is not None}
    proto_keys = list(proto_items.keys())
    proto_vals = np.vstack([proto_items[c] for c in proto_keys])
    for x in X_eval_latent:
        # compute distances to all prototypes
        dists = np.linalg.norm(proto_vals - x, axis=1)
        y_pred.append(proto_keys[np.argmin(dists)])
    return np.array(y_pred)


# ---- kNN in latent space ----
def knn_few_shot_latent(X_train_latent, y_train, X_calib_latent, y_calib, X_eval_latent, n_neighbors=3):
    X_pool = np.vstack([X_train_latent, X_calib_latent])
    y_pool = np.hstack([y_train, y_calib])
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_pool, y_pool)
    return knn.predict(X_eval_latent)


# ---- Fine-tune encoder + linear head on K-shot (very small LR, weight decay) ----
def finetune_encoder_on_calib(model, X_calib, y_calib, device="cpu",
                              epochs=30, lr=1e-3, weight_decay=1e-4,
                              finetune_encoder=True, batch_size=32, verbose=False):
    """
    Finetune model.encoder + a small linear head on calibration data.
    - X_calib, y_calib are numpy arrays
    - If finetune_encoder is False, only trains linear head (fast + safer).
    Returns: new_model (in place), and training history dict.
    """
    model.to(device)
    model.train()

    # small classification head on top of encoder
    # hidden_dim = list(model.encoder.net.children())[-2].out_features if hasattr(model.encoder, "net") else None

    linear_layers = [l for l in model.encoder.net.children() if isinstance(l, nn.Linear)]
    if len(linear_layers) == 0:
        raise ValueError("No linear layers found in encoder")
    hidden_dim = linear_layers[-1].out_features

    # Instead of trying to introspect, add a new head using encoder output size by probing one sample:
    with torch.no_grad():
        probe = torch.tensor(X_calib[:1], dtype=torch.float32).to(device)
        model.encoder.eval() 
        h_probe = model.encoder(probe)
        feat_dim = h_probe.shape[1]

    num_classes = int(np.max(y_calib)) + 1 if len(y_calib)>0 else 0
    head = nn.Linear(feat_dim, num_classes).to(device)

    # optimizer
    params = list(head.parameters())
    if finetune_encoder:
        params = list(model.encoder.parameters()) + params

    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    # dataset
    Xc = torch.tensor(X_calib, dtype=torch.float32).to(device)
    yc = torch.tensor(y_calib, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(Xc, yc)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": []}
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            feats = model.encoder(xb)  # (B,feat_dim)
            logits = head(feats)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        history["loss"].append(epoch_loss)
        if verbose:
            print(f"Finetune epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    # return head (torch module) — you can evaluate combined by passing encoder->head
    return head, history


# ---- Convenience wrapper that runs few-shot in latent space and optional finetune ----
def few_shot_on_dann(model, X_train, y_train, X_test, y_test,
                     few_shot_K=5, n_neighbors=3, alpha=0.5,
                     finetune=False, ft_epochs=30, ft_lr=1e-3,
                     device="cpu"):
    """
    Runs prototype and kNN few-shot on DANN latent space. Optionally fine-tunes encoder+head on calibration set.
    Returns dict with keys:
      - proto_before (accuracy), knn_before (accuracy)
      - proto_after (accuracy), knn_after (accuracy)   # after finetune if finetune=True
      - head (if finetune) torch module for encoder->head
    """
    # 1) get latents
    X_train_latent = get_latents(model, X_train, device=device)
    X_test_latent  = get_latents(model, X_test, device=device)

    # 2) pick K-shot per class from test subject (calibration)
    class_idxs = defaultdict(list)
    for i, g in enumerate(y_test):
        class_idxs[g].append(i)

    calib_idxs = []
    for g, idxs in class_idxs.items():
        if len(idxs) <= few_shot_K:
            calib_idxs.extend(idxs)
        else:
            calib_idxs.extend(list(np.random.choice(idxs, few_shot_K, replace=False)))
    calib_idxs = sorted(calib_idxs)

    if len(calib_idxs) == 0:
        raise ValueError("No calibration samples found for any class (K too large or test set empty).")

    X_calib = X_test[calib_idxs]
    y_calib = y_test[calib_idxs]

    eval_idxs = [i for i in range(len(y_test)) if i not in calib_idxs]
    X_eval = X_test[eval_idxs]
    y_eval = y_test[eval_idxs]

    X_calib_latent = X_test_latent[calib_idxs]
    X_eval_latent  = X_test_latent[eval_idxs]

    results = {}

    # before finetune: prototypes and knn in latent space
    y_pred_proto_before = prototype_few_shot_latent(X_train_latent, y_train, X_calib_latent, y_calib, X_eval_latent, alpha=alpha)
    results['proto_before_acc'] = np.mean(y_pred_proto_before == y_eval)

    y_pred_knn_before = knn_few_shot_latent(X_train_latent, y_train, X_calib_latent, y_calib, X_eval_latent, n_neighbors=n_neighbors)
    results['knn_before_acc'] = np.mean(y_pred_knn_before == y_eval)

    # optional finetune encoder+head on calib (in original feature space X_calib, not latent)
    results['head'] = None
    if finetune:
        # finetune on original X_calib (not latent) using encoder + new head
        head, history = finetune_encoder_on_calib(model, X_calib, y_calib, device=device,
                                                  epochs=ft_epochs, lr=ft_lr,
                                                  finetune_encoder=True, verbose=False)
        results['finetune_history'] = history
        results['head'] = head

        # evaluate after finetune: compute new latents for eval set
        X_train_latent_ft = get_latents(model, X_train, device=device)
        X_calib_latent_ft = get_latents(model, X_calib, device=device)
        X_eval_latent_ft  = get_latents(model, X_eval, device=device)

        # prototypes/knn on finetuned latent space
        y_pred_proto_after = prototype_few_shot_latent(X_train_latent_ft, y_train, X_calib_latent_ft, y_calib, X_eval_latent_ft, alpha=alpha)
        results['proto_after_acc'] = np.mean(y_pred_proto_after == y_eval)

        y_pred_knn_after = knn_few_shot_latent(X_train_latent_ft, y_train, X_calib_latent_ft, y_calib, X_eval_latent_ft, n_neighbors=n_neighbors)
        results['knn_after_acc'] = np.mean(y_pred_knn_after == y_eval)

    return results

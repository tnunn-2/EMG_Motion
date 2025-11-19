# DANN.py (updated)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from tqdm import tqdm


# -------------------------
# CNN Encoder (windowed)
# -------------------------
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=70, window_size=20, latent_dim=128):
        """
        input_channels: number of channels/features (70)
        window_size: number of frames in each window (e.g. 20)
        latent_dim: output latent dimension
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.window_size = window_size
        self.latent_dim = latent_dim
        self.fc = nn.Linear(64 * window_size, latent_dim)

    def forward(self, x):
        """
        x: (B, C=input_channels, T=window_size)
        returns: (B, latent_dim)
        """
        # print("Input to encoder:", x.shape)
        z = self.conv(x)                 # (B, 64, T)
        # print("After conv:", z.shape)
        z = z.reshape(z.size(0), -1)     # (B, 64*T)
        # print("Flattened size:", z.shape)
        z = self.fc(z)                   # (B, latent_dim)
        return z

# -------------------------
# Gesture classifier (linear head)
# -------------------------
class GestureClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=12):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h):
        return self.fc(h)

# -------------------------
# CORAL loss (covariance alignment)
# -------------------------
def coral_loss(source, target):
    """
    CORAL loss between source and target feature matrices.
    source: (B_s, D)
    target: (B_t, D)
    returns scalar loss
    """
    # Convert to float tensors
    source = source.float()
    target = target.float()

    bs = source.size(0)
    bt = target.size(0)
    d = source.size(1)

    # Center the features
    src_mean = torch.mean(source, dim=0, keepdim=True)
    tgt_mean = torch.mean(target, dim=0, keepdim=True)
    src_c = source - src_mean
    tgt_c = target - tgt_mean

    # Covariance matrices (unbiased estimator)
    # (D, D) matrices
    if bs > 1:
        cov_src = (src_c.t() @ src_c) / (bs - 1)
    else:
        cov_src = torch.zeros((d, d), device=source.device, dtype=source.dtype)

    if bt > 1:
        cov_tgt = (tgt_c.t() @ tgt_c) / (bt - 1)
    else:
        cov_tgt = torch.zeros((d, d), device=target.device, dtype=target.dtype)

    loss = torch.mean((cov_src - cov_tgt) ** 2)
    # normalize as in many implementations
    loss = loss / (4 * (d ** 2))
    return loss

# -------------------------
# DANN wrapper (Encoder + Classifier)
# -------------------------
class DANN(nn.Module):
    def __init__(self, input_channels=70, window_size=20, hidden_dim=128, num_classes=12):
        super().__init__()
        self.encoder = CNNEncoder(input_channels, window_size, latent_dim=hidden_dim)
        self.classifier = GestureClassifier(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward returns logits and latent:
          - x can be (B, C, T) windowed inputs
        """
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits, h

# -------------------------
# Training loop using CORAL
# -------------------------
def train_dann(model, src_loader, tgt_loader, num_epochs=30,
               device="cpu", lambda_coral=1.0, lr=1e-3):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    tgt_iter = iter(tgt_loader)

    print("\nTraining CORAL-DANN...\n")
    for epoch in range(num_epochs):

        # Progress bar for batches
        batch_bar = tqdm(range(len(src_loader)),
                         desc=f"Epoch {epoch+1}/{num_epochs}",
                         leave=False)

        total_clf = 0.0
        total_coral = 0.0
        total_steps = 0

        for _ in batch_bar:
            try:
                xs, ys = next(iter(src_loader))
            except StopIteration:
                src_iter = iter(src_loader)
                xs, ys = next(src_iter)

            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, _ = next(tgt_iter)

            xs = xs.to(device)
            ys = ys.to(device)
            xt = xt.to(device)

            logits_s, z_s = model(xs)
            _, z_t = model(xt)

            clf_loss = ce(logits_s, ys)
            coral = coral_loss(z_s, z_t)
            loss = clf_loss + lambda_coral * coral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_clf += clf_loss.item()
            total_coral += coral.item()
            total_steps += 1

            batch_bar.set_postfix({"clf": f"{clf_loss.item():.3f}",
                                   "coral": f"{coral.item():.3f}"})

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Avg Class Loss={total_clf/total_steps:.4f} | "
              f"Avg CORAL={total_coral/total_steps:.4f}")

# -------------------------
# Utility: get encoder latent space (numpy arrays)
# -------------------------
def get_latents(model, X, device="cpu", batch_size=256):
    """
    Returns numpy array latents for input X using model.encoder
    X: numpy array
       - either (N, D) raw-frame features (old style) OR
       - (N, C, T) windowed features (preferred)
    """
    model.to(device)
    model.eval()
    latents = []
    with torch.no_grad():
        if X.ndim == 2:
            # (N, D) -> pass through encoder expecting (B, D)
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
                # If encoder is CNN (expects 3D), try to detect and reshape
                if hasattr(model.encoder, "conv"):
                    # treat D as channels and fake time=1
                    xb = xb.unsqueeze(-1)  # (B, D, 1)
                h = model.encoder(xb)
                latents.append(h.cpu().numpy())
        elif X.ndim == 3:
            # (N, C, T)
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
                h = model.encoder(xb)
                latents.append(h.cpu().numpy())
        else:
            raise ValueError("X must be 2D (N,D) or 3D (N,C,T).")
    return np.vstack(latents)

# -------------------------
# Prototype and kNN few-shot helpers (unchanged)
# -------------------------
def prototype_few_shot_latent(X_train_latent, y_train, X_calib_latent, y_calib, X_eval_latent, alpha=0.5):
    classes = np.unique(y_train)
    prototypes = {}
    for c in classes:
        mask = (y_train == c)
        if np.any(mask):
            prototypes[c] = X_train_latent[mask].mean(axis=0)
        else:
            prototypes[c] = None

    for c in np.unique(y_calib):
        maskc = (y_calib == c)
        if prototypes.get(c) is None:
            prototypes[c] = X_calib_latent[maskc].mean(axis=0)
        else:
            prototypes[c] = (1 - alpha) * prototypes[c] + alpha * X_calib_latent[maskc].mean(axis=0)

    y_pred = []
    proto_items = {c: p for c,p in prototypes.items() if p is not None}
    proto_keys = list(proto_items.keys())
    proto_vals = np.vstack([proto_items[c] for c in proto_keys])
    for x in X_eval_latent:
        dists = np.linalg.norm(proto_vals - x, axis=1)
        y_pred.append(proto_keys[np.argmin(dists)])
    return np.array(y_pred)

def knn_few_shot_latent(X_train_latent, y_train, X_calib_latent, y_calib, X_eval_latent, n_neighbors=3):
    X_pool = np.vstack([X_train_latent, X_calib_latent])
    y_pool = np.hstack([y_train, y_calib])
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_pool, y_pool)
    return knn.predict(X_eval_latent)

# -------------------------
# Fine-tune encoder + linear head on K-shot (safe probing)
# -------------------------
def finetune_encoder_on_calib(model, X_calib, y_calib, device="cpu",
                              epochs=30, lr=1e-3, weight_decay=1e-4,
                              finetune_encoder=True, batch_size=32, verbose=False):
    """
    Finetune model.encoder + a small linear head on calibration data.
    - X_calib, y_calib are numpy arrays.
    - Handles encoder that is CNN (expects 3D) or MLP (2D).
    Returns: head (nn.Module), history dict
    """
    model.to(device)
    # Save training state
    was_training_model = model.training

    # Probe to get feature dim: use encoder.eval() to avoid BN issues on a single sample
    model.encoder.eval()
    with torch.no_grad():
        probe = None
        if X_calib.ndim == 2:
            # (N, D) -> maybe MLP or we will unsqueeze
            xb = torch.tensor(X_calib[:1], dtype=torch.float32).to(device)
            if hasattr(model.encoder, "conv"):
                xb = xb.unsqueeze(-1)  # (1, D, 1)
        elif X_calib.ndim == 3:
            xb = torch.tensor(X_calib[:1], dtype=torch.float32).to(device)
        else:
            raise ValueError("X_calib must be 2D or 3D numpy array")
        h_probe = model.encoder(xb)
        feat_dim = h_probe.shape[1]

    # Restore model training state (we'll set modes properly below)
    if was_training_model:
        model.train()
    else:
        model.eval()

    # Build head
    num_classes = int(np.max(y_calib)) + 1 if len(y_calib) > 0 else 0
    head = nn.Linear(feat_dim, num_classes).to(device)

    # Setup optimizer
    params = list(head.parameters())
    if finetune_encoder:
        params = list(model.encoder.parameters()) + params

    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    # Dataset (ensure shapes appropriate for encoder)
    Xc = torch.tensor(X_calib, dtype=torch.float32).to(device)
    yc = torch.tensor(y_calib, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(Xc, yc)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train (set encoder to train if finetune_encoder True)
    if finetune_encoder:
        model.train()
    else:
        model.eval()

    history = {"loss": []}
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            # xb already has the right ndim (2 or 3) from X_calib; ensure to call encoder with expected shape
            if xb.ndim == 2 and hasattr(model.encoder, "conv"):
                xb_in = xb.unsqueeze(-1)  # (B, D, 1) -> treated as (B, C, T)
            else:
                xb_in = xb
            opt.zero_grad()
            feats = model.encoder(xb_in)  # (B,feat_dim)
            logits = head(feats)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        history["loss"].append(epoch_loss)
        if verbose:
            print(f"Finetune epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    # Restore model training/eval state
    if was_training_model:
        model.train()
    else:
        model.eval()

    return head, history

# -------------------------
# few_shot_on_dann (unchanged except it supports windowed inputs)
# -------------------------
def few_shot_on_dann(model, X_train, y_train, X_test, y_test,
                     few_shot_K=5, n_neighbors=3, alpha=0.5,
                     finetune=False, ft_epochs=30, ft_lr=1e-3,
                     device="cpu"):
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

    # optional finetune encoder+head on calib (in original feature/window space)
    results['head'] = None
    if finetune:
        head, history = finetune_encoder_on_calib(model, X_calib, y_calib, device=device,
                                                  epochs=ft_epochs, lr=ft_lr,
                                                  finetune_encoder=True, verbose=False)
        results['finetune_history'] = history
        results['head'] = head

        # evaluate after finetune: compute new latents for eval set
        X_train_latent_ft = get_latents(model, X_train, device=device)
        X_calib_latent_ft = get_latents(model, X_calib, device=device)
        X_eval_latent_ft  = get_latents(model, X_eval, device=device)

        y_pred_proto_after = prototype_few_shot_latent(X_train_latent_ft, y_train, X_calib_latent_ft, y_calib, X_eval_latent_ft, alpha=alpha)
        results['proto_after_acc'] = np.mean(y_pred_proto_after == y_eval)

        y_pred_knn_after = knn_few_shot_latent(X_train_latent_ft, y_train, X_calib_latent_ft, y_calib, X_eval_latent_ft, n_neighbors=n_neighbors)
        results['knn_after_acc'] = np.mean(y_pred_knn_after == y_eval)

    return results

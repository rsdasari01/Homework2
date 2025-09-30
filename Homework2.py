import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import math
import warnings

# -----------------------------
# User settings / dataset path
# -----------------------------
DATA_PATH = "C:/Users/rsdas/Downloads/Housing.csv"  
RANDOM_SEED = 42
TEST_SIZE = 0.20                # 80/20 split
np.random.seed(RANDOM_SEED)

# -----------------------------
# Small utility: custom scalers (no sklearn)
# -----------------------------
class StandardScalerCustom:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        # population std (ddof=0)
        self.scale_ = X.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self
    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScalerCustom:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None
    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        rng = self.max_ - self.min_
        rng[rng == 0.0] = 1.0
        self.range_ = rng
        return self
    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.min_) / self.range_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# -----------------------------
# Custom train/validation splitter
# -----------------------------
def train_val_split(X, y, test_size=0.2, random_state=None):
    X = np.array(X)
    y = np.array(y)
    m = X.shape[0]
    idx = np.arange(m)
    rng = np.random.RandomState(seed=random_state)
    rng.shuffle(idx)
    split_at = int(np.floor((1 - test_size) * m))
    train_idx = idx[:split_at]
    val_idx = idx[split_at:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

# -----------------------------
# Dataset loading and feature prep
# -----------------------------
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Place CSV in same folder or change DATA_PATH.")
    df = pd.read_csv(path)
    return df

def prepare_features(df, feature_cols):
    """
    Extract features from dataframe.
    - Convert textual yes/no or true/false to 1/0.
    - For other object dtype columns, use pandas.get_dummies (drop_first=True).
    Returns X (numpy array) and list of used feature column names.
    """
    df2 = df.copy()
    cols = feature_cols.copy()
    # map binary textual columns to 0/1
    for c in feature_cols:
        if c in df2.columns and df2[c].dtype == object:
            vals = df2[c].dropna().unique()
            lower_vals = [str(v).strip().lower() for v in vals]
            if set(lower_vals) <= {"yes", "no"} or set(lower_vals) <= {"true", "false"} or set(lower_vals) <= {"1", "0"}:
                df2[c] = df2[c].map(lambda v: 1 if str(v).strip().lower() in ("yes", "true","1") else 0)
            else:
                # one-hot encode and replace column in feature list
                dummies = pd.get_dummies(df2[c], prefix=c, drop_first=True)
                df2 = pd.concat([df2.drop(columns=[c]), dummies], axis=1)
                # replace the original in cols with the new dummies
                idx = cols.index(c)
                cols = cols[:idx] + list(dummies.columns) + cols[idx+1:]
    missing = [c for c in cols if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")
    X = df2[cols].values.astype(np.float64)
    return X, cols

def add_bias_term(X):
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])

# -----------------------------
# Loss and gradient descent (robust)
# -----------------------------
def compute_mse(y_true, y_pred):
    # numerically stable: use float64 and square then mean
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    err = y_pred - y_true
    # use np.square (works elementwise)
    sq = np.square(err, dtype=np.float64)
    # if any infinite or nan, return np.inf to mark divergence
    if not np.isfinite(sq).all():
        return float('inf')
    return float(np.mean(sq))

def gradient_descent(X, y, lr=0.01, n_iters=2000, l2_lambda=0.0,
                     grad_clip=1e6, stop_on_diverge=True, verbose=False):
    """
    Gradient descent on MSE with optional L2 regularization (ridge).
    - grad_clip: element-wise clipping threshold for gradient to avoid overflow.
    - stop_on_diverge: if True, stop if loss becomes inf/nan and return partial history.
    Returns (theta, train_losses, diverged_flag)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    m, n = X.shape
    theta = np.zeros(n, dtype=np.float64)
    train_losses = []
    diverged = False

    for it in range(n_iters):
        preds = X.dot(theta)
        loss = compute_mse(y, preds)
        # If loss already inf or nan -> diverged
        if not np.isfinite(loss):
            diverged = True
            if verbose:
                print(f"Early stop: loss became non-finite at iter {it}.")
            break
        train_losses.append(loss)

        error = preds - y  # shape (m,)
        grad = (1.0 / m) * (X.T.dot(error))  # shape (n,)

        # L2 regularization term (non-bias)
        if l2_lambda and l2_lambda > 0:
            reg = (l2_lambda / m) * theta
            reg[0] = 0.0
            grad = grad + reg

        # Clip gradient element-wise to avoid huge steps
        grad = np.clip(grad, -grad_clip, grad_clip)

        theta = theta - lr * grad

        # If theta non-finite after update, mark divergent and stop
        if not np.isfinite(theta).all():
            diverged = True
            if verbose:
                print(f"Early stop: theta became non-finite at iter {it}.")
            break

    return theta, train_losses, diverged

# -----------------------------
# Experiment runner (no sklearn) with robust handling
# -----------------------------
def run_experiment(df, feature_cols, target_col='price', scaling=None,
                   lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000,
                   l2_lambda=0.0, random_seed=RANDOM_SEED, verbose=False):
    """
    scaling: None | 'standard' | 'minmax'
    Returns dict with results and best model info.
    Automatically retries with standard scaling if all lr diverge and scaling is None.
    """
    X_all, used_features = prepare_features(df, feature_cols.copy())
    y_all = df[target_col].values.astype(np.float64).ravel()
    # split
    X_train, X_val, y_train, y_val = train_val_split(X_all, y_all, test_size=TEST_SIZE, random_state=random_seed)

    # apply scaling if requested
    scaler = None
    if scaling == 'standard':
        scaler = StandardScalerCustom()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    elif scaling == 'minmax':
        scaler = MinMaxScalerCustom()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    X_train_b = add_bias_term(X_train)
    X_val_b = add_bias_term(X_val)

    results = []
    diverged_lrs = []
    for lr in lr_list:
        theta, train_hist, diverged = gradient_descent(X_train_b, y_train,
                                                       lr=lr, n_iters=n_iters,
                                                       l2_lambda=l2_lambda, grad_clip=1e8,
                                                       stop_on_diverge=True, verbose=verbose)
        # compute val history corresponding to train_hist length
        val_hist = []
        if len(train_hist) == 0:
            # diverged immediately
            diverged = True
        else:
            # we'll recompute theta progression to get per-step val MSE safely (limited to len(train_hist))
            theta_iter = np.zeros(X_train_b.shape[1], dtype=np.float64)
            m_train = X_train_b.shape[0]
            for it in range(len(train_hist)):
                preds_train = X_train_b.dot(theta_iter)
                err = preds_train - y_train
                grad = (1.0 / m_train) * (X_train_b.T.dot(err))
                if l2_lambda and l2_lambda > 0:
                    reg = (l2_lambda / m_train) * theta_iter
                    reg[0] = 0.0
                    grad = grad + reg
                grad = np.clip(grad, -1e8, 1e8)
                theta_iter = theta_iter - lr * grad
                tr_loss = compute_mse(y_train, X_train_b.dot(theta_iter))
                val_loss = compute_mse(y_val, X_val_b.dot(theta_iter))
                # if non-finite, mark diverged and stop collecting
                if not np.isfinite(tr_loss) or not np.isfinite(val_loss):
                    diverged = True
                    break
                val_hist.append(val_loss)

        final_val_mse = val_hist[-1] if len(val_hist) > 0 and np.isfinite(val_hist[-1]) else float('inf')
        results.append({
            'lr': lr,
            'theta': theta.copy(),
            'train_losses': train_hist,
            'val_losses': val_hist,
            'final_val_mse': final_val_mse,
            'diverged': diverged
        })
        if diverged:
            diverged_lrs.append(lr)

    all_diverged = all(r['diverged'] for r in results)
    if all_diverged and scaling is None:
        warnings.warn("All learning rates diverged with no scaling. Retrying with standardization.")
        return run_experiment(df, feature_cols, target_col=target_col, scaling='standard',
                              lr_list=lr_list, n_iters=n_iters, l2_lambda=l2_lambda,
                              random_seed=random_seed, verbose=verbose)

    # pick best non-diverging result (lowest final_val_mse)
    non_diverging = [r for r in results if not r['diverged'] and np.isfinite(r['final_val_mse'])]
    if len(non_diverging) == 0:
        # fallback: pick run with minimum final_val_mse even if inf (so plotting doesn't blow)
        best_run = results[0]
        for r in results:
            if r['final_val_mse'] < best_run['final_val_mse']:
                best_run = r
    else:
        best_run = min(non_diverging, key=lambda rr: rr['final_val_mse'])

    summary = {
        'scaling': scaling,
        'best_lr': best_run['lr'],
        'best_val_mse': best_run['final_val_mse'],
        'used_features': used_features,
        'l2_lambda': l2_lambda,
        'diverged_lrs': diverged_lrs,
        'all_results': results
    }
    return {'results': results, 'best': best_run, 'summary': summary, 'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}

# -----------------------------
# Plot helper
# -----------------------------
def plot_train_val(train_losses, val_losses, title="Train vs Validation Loss", show_legend=True):
    # defensive: if histories are empty create placeholder arrays to avoid matplotlib error
    if train_losses is None:
        train_losses = []
    if val_losses is None:
        val_losses = []
    # If both empty, plot a simple horizontal line at NaN-safe value and annotate
    if len(train_losses) == 0 and len(val_losses) == 0:
        plt.figure(figsize=(8,5))
        plt.text(0.5, 0.5, "No valid training history (all runs diverged).", ha='center', va='center')
        plt.title(title)
        return plt.gcf()
    plt.figure(figsize=(8,5))
    # Ensure arrays are finite; if any non-finite, clip to large number for plotting
    def safe_arr(a):
        a = np.array(a, dtype=np.float64)
        if a.size == 0:
            return a
        a[~np.isfinite(a)] = np.nan
        return a
    train_arr = safe_arr(train_losses)
    val_arr = safe_arr(val_losses)
    if train_arr.size > 0:
        plt.plot(train_arr, label='Train loss')
    if val_arr.size > 0:
        plt.plot(val_arr, label='Validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title(title)
    if show_legend:
        plt.legend()
    plt.grid(True)
    return plt.gcf()

# -----------------------------
# Main orchestration for Problems 1-3 (no sklearn)
# -----------------------------
def main_all_problems(data_path):
    df = load_dataset(data_path)
    possible_price_cols = [c for c in df.columns if c.lower() in ('price','house_price','houseprice','selling_price')]
    if not possible_price_cols:
        target_col = 'price'
        if target_col not in df.columns:
            raise ValueError("Couldn't auto-detect price column. Make sure your CSV has a column named 'price' or update the script.")
    else:
        target_col = possible_price_cols[0]

    # Feature sets
    feat_1a = ['area','bedrooms','bathrooms','stories','parking']
    feat_1b = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea']

    # Normalize column names to handle case differences
    df.columns = [c.strip() for c in df.columns]
    col_map_lower = {c.lower(): c for c in df.columns}
    def map_cols(cols):
        mapped = []
        for c in cols:
            key = c.lower()
            if key in col_map_lower:
                mapped.append(col_map_lower[key])
            else:
                mapped.append(c)
        return mapped

    feat_1a_mapped = map_cols(feat_1a)
    feat_1b_mapped = map_cols(feat_1b)

    # Problem 1.a baseline
    print("Running Problem 1.a (baseline, features:", feat_1a_mapped, ")")
    p1a = run_experiment(df, feat_1a_mapped, target_col=target_col, scaling=None,
                         lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000, l2_lambda=0.0)
    if p1a['summary']['diverged_lrs']:
        print("Warning: the following learning rates diverged for 1.a:", p1a['summary']['diverged_lrs'])
    fig1 = plot_train_val(p1a['best']['train_losses'], p1a['best']['val_losses'],
                          title=f"1.a Baseline - best lr={p1a['best']['lr']}, val MSE={p1a['best']['final_val_mse']:.4f}")

    # Problem 1.b baseline
    print("Running Problem 1.b (baseline, features:", feat_1b_mapped, ")")
    p1b = run_experiment(df, feat_1b_mapped, target_col=target_col, scaling=None,
                         lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000, l2_lambda=0.0)
    if p1b['summary']['diverged_lrs']:
        print("Warning: the following learning rates diverged for 1.b:", p1b['summary']['diverged_lrs'])
    fig2 = plot_train_val(p1b['best']['train_losses'], p1b['best']['val_losses'],
                          title=f"1.b Baseline - best lr={p1b['best']['lr']}, val MSE={p1b['best']['final_val_mse']:.4f}")

    # Problem 2: scaling variations
    print("Running Problem 2.a: scaling variations for Problem 1.a features")
    p2a_std = run_experiment(df, feat_1a_mapped, target_col=target_col, scaling='standard',
                             lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000, l2_lambda=0.0)
    fig3 = plot_train_val(p2a_std['best']['train_losses'], p2a_std['best']['val_losses'],
                         title=f"2.a Standardization (1.a features) - best lr={p2a_std['best']['lr']}, val MSE={p2a_std['best']['final_val_mse']:.4f}")

    p2a_minmax = run_experiment(df, feat_1a_mapped, target_col=target_col, scaling='minmax',
                                lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000, l2_lambda=0.0)
    fig4 = plot_train_val(p2a_minmax['best']['train_losses'], p2a_minmax['best']['val_losses'],
                         title=f"2.a Normalization (1.a features) - best lr={p2a_minmax['best']['lr']}, val MSE={p2a_minmax['best']['final_val_mse']:.4f}")

    print("Running Problem 2.b: scaling variations for Problem 1.b features")
    p2b_std = run_experiment(df, feat_1b_mapped, target_col=target_col, scaling='standard',
                             lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000, l2_lambda=0.0)
    fig5 = plot_train_val(p2b_std['best']['train_losses'], p2b_std['best']['val_losses'],
                         title=f"2.b Standardization (1.b features) - best lr={p2b_std['best']['lr']}, val MSE={p2b_std['best']['final_val_mse']:.4f}")

    p2b_minmax = run_experiment(df, feat_1b_mapped, target_col=target_col, scaling='minmax',
                                lr_list=[0.01, 0.02, 0.05, 0.1], n_iters=3000, l2_lambda=0.0)
    fig6 = plot_train_val(p2b_minmax['best']['train_losses'], p2b_minmax['best']['val_losses'],
                         title=f"2.b Normalization (1.b features) - best lr={p2b_minmax['best']['lr']}, val MSE={p2b_minmax['best']['final_val_mse']:.4f}")

    # Problem 3: L2 regularization grid search for best scaling from problem 2
    print("Selecting best scaling for Problem 3 experiments...")
    best_p2a = p2a_std if p2a_std['summary']['best_val_mse'] < p2a_minmax['summary']['best_val_mse'] else p2a_minmax
    best_p2b = p2b_std if p2b_std['summary']['best_val_mse'] < p2b_minmax['summary']['best_val_mse'] else p2b_minmax
    print(f"Best scaling for 2.a features: {best_p2a['summary']['scaling']}, val MSE={best_p2a['summary']['best_val_mse']:.4f}")
    print(f"Best scaling for 2.b features: {best_p2b['summary']['scaling']}, val MSE={best_p2b['summary']['best_val_mse']:.4f}")

    l2_candidates = [0.0, 0.01, 0.1, 1.0]
    p3a_results = {}
    print("Running Problem 3.a (1.a features) with L2 regularization grid...")
    for l2 in l2_candidates:
        r = run_experiment(df, feat_1a_mapped, target_col=target_col, scaling=best_p2a['summary']['scaling'],
                           lr_list=[best_p2a['summary']['best_lr']], n_iters=3000, l2_lambda=l2)
        p3a_results[l2] = r
    best_l2_p3a = min(p3a_results.items(), key=lambda kv: kv[1]['best']['final_val_mse'])[0]
    fig7 = plot_train_val(p3a_results[best_l2_p3a]['best']['train_losses'], p3a_results[best_l2_p3a]['best']['val_losses'],
                         title=f"3.a (1.a features) L2={best_l2_p3a} - val MSE={p3a_results[best_l2_p3a]['best']['final_val_mse']:.4f}")

    print("Running Problem 3.b (1.b features) with L2 regularization grid...")
    p3b_results = {}
    for l2 in l2_candidates:
        r = run_experiment(df, feat_1b_mapped, target_col=target_col, scaling=best_p2b['summary']['scaling'],
                           lr_list=[best_p2b['summary']['best_lr']], n_iters=3000, l2_lambda=l2)
        p3b_results[l2] = r
    best_l2_p3b = min(p3b_results.items(), key=lambda kv: kv[1]['best']['final_val_mse'])[0]
    fig8 = plot_train_val(p3b_results[best_l2_p3b]['best']['train_losses'], p3b_results[best_l2_p3b]['best']['val_losses'],
                         title=f"3.b (1.b features) L2={best_l2_p3b} - val MSE={p3b_results[best_l2_p3b]['best']['final_val_mse']:.4f}")

    pdf_path = "report_no_sklearn_fixed.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        txt = []
        txt.append("Homework 2 - Linear Regression with Gradient Descent (NO sklearn) - FIXED\n\n")
        txt.append(f"Dataset: {data_path}\n\n")
        txt.append("Summary of best results (validation MSE):\n")
        txt.append(f"1.a (baseline) - val MSE: {p1a['best']['final_val_mse']:.4f}, lr={p1a['best']['lr']}\n")
        txt.append(f"1.b (baseline) - val MSE: {p1b['best']['final_val_mse']:.4f}, lr={p1b['best']['lr']}\n\n")
        txt.append("Scaling comparisons (Problem 2):\n")
        txt.append(f"2.a best scaling: {best_p2a['summary']['scaling']} val MSE: {best_p2a['summary']['best_val_mse']:.4f}\n")
        txt.append(f"2.b best scaling: {best_p2b['summary']['scaling']} val MSE: {best_p2b['summary']['best_val_mse']:.4f}\n\n")
        txt.append("Regularization (Problem 3):\n")
        txt.append(f"3.a best L2: {best_l2_p3a} val MSE: {p3a_results[best_l2_p3a]['best']['final_val_mse']:.4f}\n")
        txt.append(f"3.b best L2: {best_l2_p3b} val MSE: {p3b_results[best_l2_p3b]['best']['final_val_mse']:.4f}\n")
        plt.text(0.01, 0.99, "".join(txt), va='top', fontsize=10, family='monospace')
        pdf.savefig()
        plt.close()

    print(f"All done. Figures and summary saved into {pdf_path}.")
    print("Final summary (short):")
    print(f"1.a val MSE: {p1a['best']['final_val_mse']:.6f}, lr={p1a['best']['lr']}")
    print(f"1.b val MSE: {p1b['best']['final_val_mse']:.6f}, lr={p1b['best']['lr']}")
    print(f"2.a best scaling: {best_p2a['summary']['scaling']} val MSE: {best_p2a['summary']['best_val_mse']:.6f}")
    print(f"2.b best scaling: {best_p2b['summary']['scaling']} val MSE: {best_p2b['summary']['best_val_mse']:.6f}")
    print(f"3.a best L2: {best_l2_p3a} val MSE: {p3a_results[best_l2_p3a]['best']['final_val_mse']:.6f}")
    print(f"3.b best L2: {best_l2_p3b} val MSE: {p3b_results[best_l2_p3b]['best']['final_val_mse']:.6f}")

if __name__ == "__main__":
    main_all_problems(DATA_PATH)

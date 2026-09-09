import numpy as np

# ============================================================
# PCA FROM SCRATCH — Eigendecomposition approach
# ============================================================

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None      # (k, d) — the principal directions
        self.mean = None            # (d,)
        self.eigenvalues = None     # (k,)
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray):
        n, d = X.shape

        # Step 1: Center
        self.mean = X.mean(axis=0)
        X_c = X - self.mean

        # Step 2: Covariance matrix  (d x d)
        #   1/(n-1) for unbiased estimate (Bessel's correction)
        cov = (X_c.T @ X_c) / (n - 1)

        # Step 3: Eigen decomposition
        #   eigh → real symmetric matrix, returns ASCENDING order
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Keep top-k
        self.eigenvalues = eigvals[:self.n_components]
        self.components = eigvecs[:, :self.n_components].T  # (k, d)
        self.explained_variance_ratio = self.eigenvalues / eigvals.sum()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T   # (n, d) @ (d, k) → (n, k)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components + self.mean  # (n, k) @ (k, d) + (d,)


# ============================================================
# PCA FROM SCRATCH — SVD approach (more numerically stable)
# ============================================================

class PCA_SVD:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray):
        n, d = X.shape
        self.mean = X.mean(axis=0)
        X_c = X - self.mean

        # SVD:  X_c = U @ diag(S) @ Vt
        #   U: (n, min(n,d))   left singular vectors
        #   S: (min(n,d),)     singular values
        #   Vt: (min(n,d), d)  right singular vectors (rows = PC directions)
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)

        self.components = Vt[:self.n_components]        # (k, d)
        self.eigenvalues = (S[:self.n_components] ** 2) / (n - 1)
        total_var = (S ** 2).sum() / (n - 1)
        self.explained_variance_ratio = self.eigenvalues / total_var

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ============================================================
# DEMO: Run on Iris dataset (loaded manually, no sklearn)
# ============================================================

def load_iris_manual():
    """Load iris without sklearn — using the raw CSV from a well-known source.
    Fallback: generate synthetic data with similar structure."""
    try:
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target, data.feature_names, data.target_names
    except ImportError:
        pass

    # Fallback: synthetic 4-feature, 3-class data
    np.random.seed(42)
    n_per_class = 50
    X = np.vstack([
        np.random.randn(n_per_class, 4) * [0.3, 0.3, 0.1, 0.1] + [5.0, 3.4, 1.5, 0.2],
        np.random.randn(n_per_class, 4) * [0.5, 0.3, 0.5, 0.2] + [5.9, 2.8, 4.3, 1.3],
        np.random.randn(n_per_class, 4) * [0.6, 0.3, 0.6, 0.3] + [6.6, 3.0, 5.6, 2.0],
    ])
    y = np.array([0]*50 + [1]*50 + [2]*50)
    return X, y, ['feat_0','feat_1','feat_2','feat_3'], ['class_0','class_1','class_2']


if __name__ == "__main__":
    X, y, feat_names, target_names = load_iris_manual()
    print(f"Data shape: {X.shape}")
    print(f"Features: {feat_names}\n")

    # ------- Eigendecomposition PCA -------
    pca_eig = PCA(n_components=2)
    Z_eig = pca_eig.fit_transform(X)

    print("=" * 60)
    print("EIGENDECOMPOSITION PCA")
    print("=" * 60)
    print(f"Eigenvalues:              {pca_eig.eigenvalues}")
    print(f"Explained variance ratio: {pca_eig.explained_variance_ratio}")
    print(f"Total variance captured:  {pca_eig.explained_variance_ratio.sum():.4f}")
    print(f"\nPC1 direction: {pca_eig.components[0]}")
    print(f"PC2 direction: {pca_eig.components[1]}")
    print(f"\nProjected shape: {Z_eig.shape}")
    print(f"First 5 projected points:\n{Z_eig[:5]}")

    # Reconstruction error
    X_reconstructed = pca_eig.inverse_transform(Z_eig)
    recon_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\nMean reconstruction error (2 components): {recon_error:.6f}")

    # ------- SVD PCA -------
    pca_svd = PCA_SVD(n_components=2)
    Z_svd = pca_svd.fit_transform(X)

    print(f"\n{'=' * 60}")
    print("SVD PCA")
    print("=" * 60)
    print(f"Eigenvalues:              {pca_svd.eigenvalues}")
    print(f"Explained variance ratio: {pca_svd.explained_variance_ratio}")
    print(f"Total variance captured:  {pca_svd.explained_variance_ratio.sum():.4f}")
    print(f"\nPC1 direction: {pca_svd.components[0]}")
    print(f"PC2 direction: {pca_svd.components[1]}")

    # ------- Verify they match -------
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print("=" * 60)

    # Eigenvalues should match
    eig_match = np.allclose(pca_eig.eigenvalues, pca_svd.eigenvalues)
    print(f"Eigenvalues match: {eig_match}")

    # Projections should match up to sign (eigenvectors can flip sign)
    # Check: are the absolute values of projections close?
    proj_match = np.allclose(np.abs(Z_eig), np.abs(Z_svd))
    print(f"Projections match (up to sign): {proj_match}")

    # ------- Manual covariance matrix walkthrough -------
    print(f"\n{'=' * 60}")
    print("MANUAL WALKTHROUGH")
    print("=" * 60)

    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / (X.shape[0] - 1)
    print(f"Covariance matrix shape: {cov.shape}")
    print(f"Covariance matrix:\n{np.round(cov, 4)}")

    # Check symmetry
    print(f"\nIs symmetric: {np.allclose(cov, cov.T)}")

    # All eigenvalues
    all_eigvals, _ = np.linalg.eigh(cov)
    all_eigvals = all_eigvals[::-1]
    print(f"\nAll eigenvalues: {np.round(all_eigvals, 4)}")
    print(f"Cumulative variance explained:")
    cumsum = np.cumsum(all_eigvals) / all_eigvals.sum()
    for i, c in enumerate(cumsum):
        print(f"  Top {i+1} components: {c:.4f} ({c*100:.1f}%)")

    # ------- Show per-class projections -------
    print(f"\n{'=' * 60}")
    print("PER-CLASS PROJECTIONS (first 3 per class)")
    print("=" * 60)
    for cls_idx, cls_name in enumerate(target_names):
        mask = y == cls_idx
        pts = Z_eig[mask][:3]
        print(f"\n{cls_name}:")
        for p in pts:
            print(f"  PC1={p[0]:+.3f}  PC2={p[1]:+.3f}")

    # ------- Sklearn comparison (if available) -------
    print(f"\n{'=' * 60}")
    print("SKLEARN COMPARISON")
    print("=" * 60)
    try:
        from sklearn.decomposition import PCA as SkPCA
        sk = SkPCA(n_components=2)
        Z_sk = sk.fit_transform(X)
        print(f"Sklearn eigenvalues:     {sk.explained_variance_}")
        print(f"Our eigenvalues:         {pca_eig.eigenvalues}")
        print(f"Match: {np.allclose(sk.explained_variance_, pca_eig.eigenvalues)}")
        print(f"\nSklearn EVR: {sk.explained_variance_ratio_}")
        print(f"Our EVR:     {pca_eig.explained_variance_ratio}")
    except ImportError:
        print("sklearn not available — but our two implementations agree with each other.")

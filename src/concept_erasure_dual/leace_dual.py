# The code adapted from https://github.com/EleutherAI/concept-erasure/blob/main/concept_erasure/leace.py
import numpy as np
import torch
from torch import Tensor

from dataclasses import dataclass

from typing import Tuple


def optimal_linear_shrinkage(
        S_n: Tensor, n: int | Tensor, *, inplace: bool = False
) -> Tensor:
    """Optimal linear shrinkage for a sample covariance matrix or batch thereof.

    Given a sample covariance matrix `S_n` of shape (*, p, p) and a sample size `n`,
    this function computes the optimal shrinkage coefficients `alpha` and `beta`, then
    returns the covariance estimate `alpha * S_n + beta * Sigma0`, where `Sigma0` is
    an isotropic covariance matrix with the same trace as `S_n`.

    The formula is distribution-free and asymptotically optimal in the Frobenius norm
    among all linear shrinkage estimators as the dimensionality `p` and sample size `n`
    jointly tend to infinity, with the ratio `p / n` converging to a finite positive
    constant `c`. The derivation is based on Random Matrix Theory and assumes that the
    underlying distribution has finite moments up to 4 + eps, for some eps > 0.

    See "On the Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
    Dimensional Covariance Matrix" <https://arxiv.org/abs/1308.2608> for details.

    Args:
        S_n: Sample covariance matrices of shape (*, p, p).
        n: Sample size.
    """
    p = S_n.shape[-1]
    assert S_n.shape[-2:] == (p, p)

    trace_S = trace(S_n)

    # Since sigma0 is I * tr(S_n) / p, its squared Frobenius norm is tr(S_n) ** 2 / p.
    sigma0_norm_sq = trace_S ** 2 / p
    S_norm_sq = S_n.norm(dim=(-2, -1), keepdim=True) ** 2

    prod_trace = sigma0_norm_sq
    top = trace_S * trace_S.conj() * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace * prod_trace.conj()

    # Epsilon prevents dividing by zero for the zero matrix. In that case we end up
    # setting alpha = 0, beta = 1, but it doesn't matter since we're shrinking toward
    # tr(0)*I = 0, so it's a no-op.
    eps = torch.finfo(S_n.dtype).eps
    alpha = 1 - (top + eps) / (bottom + eps)
    beta = (1 - alpha) * (prod_trace + eps) / (sigma0_norm_sq + eps)

    ret = S_n.mul_(alpha) if inplace else alpha * S_n
    diag = beta * trace_S / p
    torch.linalg.diagonal(ret).add_(diag.squeeze(-1))
    return ret


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)


@dataclass(frozen=True)
class LeaceEraserDual:
    """LEACE eraser that surgically erases a concept from a representation.

    Since the LEACE projection matrix is guaranteed to be a rank k - 1 perturbation of
    the identity, we store it implicitly in the d x k matrices `proj_left` and
    `proj_right`. The full matrix is given by `torch.eye(d) - proj_left @ proj_right`.
    """

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None

    @classmethod
    def fit(cls, xb: Tensor, xf: Tensor, b: Tensor, f: Tensor, **kwargs) -> "LeaceEraserDual":
        """Convenience method to fit a LeaceEraser on data and return it."""
        return LeaceFitterDual.fit(xb, xf, b, f, **kwargs).eraser

    @classmethod
    def variance_properties(cls, xb: Tensor, xf: Tensor, b: Tensor, f: Tensor, **kwargs) -> Tuple:
        """Returns the singular values and variance of the bias and feature parts."""
        return LeaceFitterDual.fit(xb, xf, b, f, **kwargs).variance_properties()

    @property
    def P(self) -> Tensor:
        """The projection matrix."""
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the projection to the input tensor."""
        delta = x - self.bias if self.bias is not None else x

        # Ensure we do the matmul in the most efficient order.
        x_ = x - (delta @ self.proj_right.mH) @ self.proj_left.mH
        return x_.type_as(x)

    def to(self, device: torch.device | str) -> "LeaceEraserDual":
        """Move eraser to a new device."""
        return LeaceEraserDual(
            self.proj_left.to(device),
            self.proj_right.to(device),
            self.bias.to(device) if self.bias is not None else None,
        )


class LeaceFitterDual:
    """Fits an affine transform that surgically erases a concept from a representation.

    This class implements Least-squares Concept Erasure (LEACE) from
    https://arxiv.org/abs/2306.03819.

    This class stores all the covariance statistics needed to compute the LEACE eraser.
    This allows the statistics to be updated incrementally with `update()`.
    """

    mean_x: Tensor
    """Running mean of X."""

    mean_b: Tensor
    """Running mean of Z."""

    mean_f: Tensor
    """Running mean of F."""

    sigma_xb_: Tensor
    """Unnormalized cross-covariance matrix X^T B."""

    sigma_xf_: Tensor
    """Unnormalized cross-covariance matrix X^T F."""

    sigma_xx_: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    nb: Tensor
    """Number of X samples seen so far with bias variable."""

    nf: Tensor
    """Number of X samples seen so far with feature variable."""

    @classmethod
    def fit(cls, xb: Tensor, xf: Tensor, b: Tensor, f: Tensor, ** kwargs) -> "LeaceFitterDual":
        """Convenience method to fit a LeaceFitter on data and return it."""
        nb, db = xb.shape
        _, kb = b.reshape(nb, -1).shape
        nf, df = xf.shape
        _, kf = f.reshape(nf, -1).shape

        assert db == df, f"The rows of X's must have the same number of features."
        d = db

        fitter = LeaceFitterDual(d, kb, kf, device=xb.device, dtype=xb.dtype, **kwargs)
        fitter = fitter.update(xb, b, update_type="b")
        fitter = fitter.update(xf, f, update_type="f")

        return fitter

    def __init__(
            self,
            x_dim: int,
            b_dim: int,
            f_dim: int,
            *,
            affine: bool = True,
            constrain_cov_trace: bool = True,
            device: str | torch.device | None = None,
            dtype: torch.dtype | None = None,
            shrinkage: bool = True,
            svd_tol: float = 0.01,
            b_to_f_ratio_tol: float = None,
    ):
        """Initialize a `LeaceFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            b_dim: Dimensionality of the bias concept to nullify.
            f_dim: Dimensionality of the feature concept to preserve.
            affine: Whether to use a bias term to ensure the unconditional mean of the
                features remains the same after erasure.
            constrain_cov_trace: Whether to constrain the trace of the covariance of X
                after erasure to be no greater than before erasure. This is especially
                useful when injecting the scrubbed features back into a model. Without
                this constraint, the norm of the model's hidden states may diverge in
                some cases.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
            svd_tol: Singular values under this threshold are truncated, both during
                the phase where we do SVD on the cross-covariance matrix, and at the
                phase where we compute the pseudoinverse of the projected covariance
                matrix. Higher values are more numerically stable and result in less
                damage to the representation, but may leave trace correlations intact.
            b_to_f_ratio_tol: The ratio of bias to feature variance above which the
                dimension is nullified. If None, b_dim / f_dim is assumed.
        """
        super().__init__()

        self.x_dim = x_dim
        self.b_dim = b_dim
        self.f_dim = f_dim

        self.affine = affine
        self.constrain_cov_trace = constrain_cov_trace
        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol
        if b_to_f_ratio_tol is None:
            self.b_to_f_ratio_tol = b_dim / f_dim
        else:
            assert b_to_f_ratio_tol > 0.0, "The ratio must be positive."
            self.b_to_f_ratio_tol = b_to_f_ratio_tol

        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.mean_b = torch.zeros(b_dim, device=device, dtype=dtype)
        self.mean_f = torch.zeros(f_dim, device=device, dtype=dtype)

        self.nb = torch.tensor(0, device=device)
        self.nf = torch.tensor(0, device=device)
        self.sigma_xb_ = torch.zeros(x_dim, b_dim, device=device, dtype=dtype)
        self.sigma_xf_ = torch.zeros(x_dim, f_dim, device=device, dtype=dtype)
        self.sigma_xx_ = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)


    @torch.no_grad()
    def update(self, x: Tensor, z: Tensor, update_type: str) -> "LeaceFitterDual":
        """Update the running statistics with a new batch of data."""

        if update_type == "b":
            sigma_xz_ = self.sigma_xb_
            mean_z = self.mean_b
        elif update_type == "f":
            sigma_xz_ = self.sigma_xf_
            mean_z = self.mean_f
        else:
            raise ValueError(f"Unknown update type: {update_type}. Should be 'b' or 'f'.")

        d, c = sigma_xz_.shape
        x = x.reshape(-1, d).type_as(self.mean_x)
        n, _ = x.shape

        if update_type == "b":
            self.nb += n
        elif update_type == "f":
            self.nf += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / (self.nb + self.nf)
        delta_x2 = x - self.mean_x

        # Update the covariance matrix of X if needed (for LEACE)
        assert self.sigma_xx_ is not None
        self.sigma_xx_.addmm_(delta_x.mH, delta_x2)

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - mean_z
        mean_z += delta_z.sum(dim=0) / (self.nb if update_type == "b" else self.nf)
        delta_z2 = z - mean_z

        # Update the cross-covariance matrix
        sigma_xz_.addmm_(delta_x.mH, delta_z2)

        return self

    @property
    def eraser(self) -> LeaceEraserDual:
        """Erasure function lazily computed given the current statistics."""
        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)

        # Compute the whitening and unwhitening matrices

        sigma = self.sigma_xx
        L, V = torch.linalg.eigh(sigma)

        # Threshold used by torch.linalg.pinv
        mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)

        # Assuming PSD; account for numerical error
        L.clamp_min_(0.0)

        W = V * torch.where(mask, L.rsqrt(), 0.0) @ V.mH
        W_inv = V * torch.where(mask, L.sqrt(), 0.0) @ V.mH


        sigma_xbf = torch.cat([self.sigma_xb, self.sigma_xf], dim=1)
        u, s, vh = torch.linalg.svd(W @ sigma_xbf, full_matrices=False)

        b_var_part = vh[:,:self.b_dim].square().sum(dim=1, keepdim=False)
        f_var_part = vh[:,-self.f_dim:].square().sum(dim=1, keepdim=False)


        # Throw away singular values that are too small
        u *= s > self.svd_tol

        # Nullify the bias dimension only if it is too large relative to the feature dimension
        u *= ( b_var_part / f_var_part > self.b_to_f_ratio_tol )


        proj_left = W_inv @ u
        proj_right = u.mH @ W


        if self.constrain_cov_trace:
            P = eye - proj_left @ proj_right

            # Prevent the covariance trace from increasing
            sigma = self.sigma_xx
            old_trace = torch.trace(sigma)
            new_trace = torch.trace(P @ sigma @ P.mH)

            # If applying the projection matrix increases the variance, this might
            # cause instability, especially when erasure is applied multiple times.
            # We regularize toward the orthogonal projection matrix to avoid this.
            if new_trace.real > old_trace.real:
                Q = eye - u @ u.mH

                # Set up the variables for the quadratic equation
                x = new_trace
                y = 2 * torch.trace(P @ sigma @ Q.mH)
                z = torch.trace(Q @ sigma @ Q.mH)
                w = old_trace

                # Solve for the mixture of P and Q that makes the trace equal to the
                # trace of the original covariance matrix
                discr = torch.sqrt(
                    4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y ** 2
                )
                alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
                alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

                # Choose the positive root
                alpha = torch.where(alpha1.real > 0, alpha1, alpha2).clamp(0, 1)
                P = alpha * P + (1 - alpha) * Q

                # TODO: Avoid using SVD here
                u, s, vh = torch.linalg.svd(eye - P)
                proj_left = u * s.sqrt()
                proj_right = vh * s.sqrt()

        return LeaceEraserDual(
            proj_left, proj_right, bias=self.mean_x if self.affine else None
        )

    def variance_properties(self) -> Tuple:
        # Compute the whitening and unwhitening matrices

        sigma = self.sigma_xx
        L, V = torch.linalg.eigh(sigma)

        # Threshold used by torch.linalg.pinv
        mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)

        # Assuming PSD; account for numerical error
        L.clamp_min_(0.0)

        W = V * torch.where(mask, L.rsqrt(), 0.0) @ V.mH

        sigma_xbf = torch.cat([self.sigma_xb, self.sigma_xf], dim=1)
        u, s, vh = torch.linalg.svd(W @ sigma_xbf, full_matrices=False)

        b_var_part = vh[:,:self.b_dim].square().sum(dim=1, keepdim=False)
        f_var_part = vh[:,-self.f_dim:].square().sum(dim=1, keepdim=False)


        return u.cpu().numpy(), s.cpu().numpy(), b_var_part.cpu().numpy(), f_var_part.cpu().numpy()
    @property
    def sigma_xx(self) -> Tensor:
        """The covariance matrix of X."""
        assert self.nb + self.nf > 1, "Call update() before accessing sigma_xx"
        assert (
                self.sigma_xx_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_xx_ + self.sigma_xx_.mH) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / (self.nb + self.nf) , self.nb + self.nf, inplace=True)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.nb + self.nf - 1)

    @property
    def sigma_xb(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.nb > 1, "Call update() with labels before accessing sigma_xb"
        return self.sigma_xb_ / (self.nb - 1)

    @property
    def sigma_xf(self) -> Tensor:
        assert self.nf > 1, "Call update() with labels before accessing sigma_xf"
        return self.sigma_xf_ / (self.nf - 1)
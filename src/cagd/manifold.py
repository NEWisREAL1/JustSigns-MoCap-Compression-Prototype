from abc import ABC, abstractmethod

import numpy as np


class IManifoldSpace(ABC):
    """
    Interface for manifolds (domain-specific) classes
    """

    @abstractmethod
    def distance(self, p1, p2):
        """Calculate geodesic distance between vectors"""
        pass
    
    @abstractmethod
    def evaluate_curve(self, t_array, control_pts, collocation_mat):
        """Evaluating B-Spline at given time values"""
        pass
    
    @abstractmethod
    def calculate_error(self, approx_pts, target_pts):
        """Calculate fitting error"""
        pass
    
    @abstractmethod
    def apply_update(self, control_pts, update_vecs):
        """Update control points based on given update vector"""
        pass
    
    @abstractmethod
    def interpolate(self, p1, p2, alphas):
        """Geodesic interpolation between points"""
        pass


class EuclideanSpace(IManifoldSpace):
    """
    Encapsulator for mathematical opaertions specific to Euclidean space
    """

    def distance(self, p1, p2):
        p1 = np.atleast_2d(p1)
        p2 = np.atleast_2d(p2)
        return np.linalg.norm(p1 - p2, axis=1)
        
    def evaluate_curve(self, control_pts, collocation_mat):
        control_pts = np.atleast_2d(control_pts)
        return collocation_mat @ control_pts
    
    def calculate_error(self, approx_pts, target_pts):
        approx_pts = np.atleast_2d(approx_pts)
        target_pts = np.atleast_2d(target_pts)
        return target_pts - approx_pts
    
    def apply_update(self, control_pts, update_vecs):
        control_pts = np.atleast_2d(control_pts)
        update_vecs = np.atleast_2d(update_vecs)
        return control_pts + update_vecs

    def interpolate(self, p1, p2, alphas):
        p1 = np.atleast_2d(p1)
        p2 = np.atleast_2d(p2)
        alphas = np.atleast_2d(alphas)
        return ((1 - alphas) * p1) + (alphas * p2)


class QuaternionSpace:
    """
    Encapsulator for mathematical operations specific to the S3 Quaternion manifold.
    Assumes quaternion format: [x, y, z, w]
    """

    def distance(self, p1, p2):
        p1 = np.atleast_2d(p1)
        p2 = np.atleast_2d(p2)
        dots = np.sum(p1 * p2, axis=1)
        clipped_dot = np.clip(np.abs(dots), 0.0, 1.0)
        return 2.0 * np.arccos(clipped_dot)
        
    def evaluate_curve(self, control_pts, collocation_mat):
        control_pts = np.atleast_2d(control_pts)
        m, n = collocation_mat.shape
        cumulative_basis = np.cumsum(collocation_mat[:, ::-1], axis=1)[:, ::-1]

        p_prevs = control_pts[:-1]
        p_currs = control_pts[1:]
        delta_qs = self.quat_multiply(self.quat_inverse(p_prevs), p_currs)
        omegas = self.quat_log(delta_qs) # Shape: (n-1, 3)
            
        scaled_omegas = cumulative_basis[:, 1:, np.newaxis] * omegas[np.newaxis, :, :]
        
        # Flatten for batched quat_exp, then reshape back
        flat_omegas = scaled_omegas.reshape(-1, 3)
        flat_step_quats = self.quat_exp(flat_omegas)
        step_quats = flat_step_quats.reshape(m, n - 1, 4)
        
        curve_evals = np.tile(control_pts[0], (m, 1))
        
        # Sequential accumulation (Cannot be parallelized due to non-commutative multiplication)
        for i in range(n - 1):
            curve_evals = self.quat_multiply(curve_evals, step_quats[:, i, :])
            
        # Safety: Force Normalize
        norms = np.linalg.norm(curve_evals, axis=1, keepdims=True)
        return curve_evals / norms
    
    def calculate_error(self, approx_pts, target_pts):
        approx_pts = np.atleast_2d(approx_pts)
        target_pts = np.atleast_2d(target_pts)
        sph_err = self.quat_multiply(self.quat_inverse(approx_pts), target_pts)
        return self.quat_log(sph_err)
    
    def apply_update(self, control_pts, update_vecs):
        control_pts = np.atleast_2d(control_pts)
        update_vecs = np.atleast_2d(update_vecs)
        updated_cps = self.quat_multiply(control_pts, self.quat_exp(update_vecs))
        
        norms = np.linalg.norm(updated_cps, axis=1, keepdims=True)
        return updated_cps / norms

    def interpolate(self, p1, p2, alphas):
        p1, p2, alphas = np.atleast_2d(p1), np.atleast_2d(p2), np.atleast_2d(alphas)
        
        dots = np.sum(p1 * p2, axis=1, keepdims=True)
        
        # Ensure shortest path on the sphere
        p2_adj = np.where(dots < 0, -p2, p2)
        dots = np.abs(dots)
        
        dots = np.clip(dots, -1.0, 1.0)
        thetas = np.arccos(dots)
        sins = np.sin(thetas)
        
        res = np.empty_like(p1)
        safe = (sins > 1e-8).flatten()
        
        # SLERP for large angles
        w1 = np.sin((1.0 - alphas[safe]) * thetas[safe]) / sins[safe]
        w2 = np.sin(alphas[safe] * thetas[safe]) / sins[safe]
        res[safe] = w1 * p1[safe] + w2 * p2_adj[safe]
        
        # LERP for tiny angles (to avoid div by zero)
        res[~safe] = (1.0 - alphas[~safe]) * p1[~safe] + alphas[~safe] * p2_adj[~safe]
        
        norms = np.linalg.norm(res, axis=1, keepdims=True)
        return res / norms

    # ----- General Utils ----- #
    
    def quat_conjugate(self, q):
        q = np.atleast_2d(q)
        q_conj = np.empty_like(q)
        q_conj[:, -1] =  q[:, -1]
        q_conj[:, :-1] = -q[:, :-1]
        return q_conj

    def quat_inverse(self, q):
        q = np.atleast_2d(q)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        return self.quat_conjugate(q) / (norms ** 2)

    def quat_multiply(self, q1, q2):
        q1 = np.atleast_2d(q1)
        q2 = np.atleast_2d(q2)

        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        res = np.empty_like(q1)
        res[:, 0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        res[:, 1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        res[:, 2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        res[:, 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return res

    def quat_log(self, q):
        """Map quaternion to tangent space with singularity protection"""
        q = np.atleast_2d(q)
        w = np.clip(q[:, -1], -1.0, 1.0)
        thetas = np.arccos(w)
        sins = np.sin(thetas)
        
        res = np.empty((q.shape[0], 3))
        safe = sins > 1e-8
        
        # If angle is large enough, map normally
        res[safe] = (thetas[safe, np.newaxis] / sins[safe, np.newaxis]) * q[safe, :-1]
        # If angle is near 0, the tangent vector approaches the (x,y,z) components
        res[~safe] = q[~safe, :-1]
        return res

    def quat_exp(self, v):
        """Map tangent vector to quaternion space with singularity protection"""
        v = np.atleast_2d(v)
        norms = np.linalg.norm(v, axis=1, keepdims=True)

        q = np.empty((v.shape[0], 4))
        safe = (norms > 1e-8).flatten()
        
        q[safe, -1] = np.cos(norms[safe]).flatten()
        q[safe, :-1] = (np.sin(norms[safe]) / norms[safe]) * v[safe]
        
        # If vector is zero-length, return identity quaternion
        q[~safe, -1] = 1.0
        q[~safe, :-1] = 0.0
        return q






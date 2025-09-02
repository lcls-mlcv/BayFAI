import numpy as np
import pyFAI
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from mpi4py import MPI
from time import time

from bayfai.geometry import calculate_2theta

pyFAI.use_opencl = False

class BayesGeomOpt:
    """
    Class to perform Geometry Optimization using Bayesian Optimization wrapped over PyFAI

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detector : PyFAI.Detector
        PyFAI detector object
    powder : np.ndarray
        Powder pattern data
    calibrant : PyFAI.Calibrant
        Calibrant object
    fixed : list
        List of parameters to keep fixed during optimization
    """

    def __init__(
        self,
        exp,
        run,
        detector,
        powder,
        calibrant,
        fixed,
    ):
        self.exp = exp
        self.run = run
        self.det_name = detector.name
        self.detector = detector
        self.powder = powder
        self.stacked_powder = np.reshape(powder, detector.shape)
        self.calibrant = calibrant
        self.fixed = fixed
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.tth = np.array(calibrant.get_2th())
        self.space = []
        for p in self.order:
            if p not in self.fixed:
                self.space.append(p)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    @staticmethod
    def UCB(X, gp_model, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        next = np.argmax(ucb)
        return next

    @staticmethod
    def q_UCB(X, gp_model, q, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        top_next = np.argsort(ucb)[-q:]
        return top_next

    def create_search_space(self, bounds, center, res):
        """
        Dynamically discretize the search space based on bounds.
        
        Parameters
        ----------
        bounds : dict
            Bounds for each parameter, format: {param: (lower, upper)}
        center : dict
            Center values for each parameter
        res : dict
            Resolution per parameter
        
        Returns
        -------
        X : np.ndarray
            Full 6D geometry space (cartesian product)
        X_norm : np.ndarray
            Normalized search space (between-1 and 1)
        """
        full_params = {}
        search_params = {}
        for p in self.order:
            if p in self.space:
                low = center[p] + bounds[p][0]
                high = center[p] + bounds[p][1]
                step = res[p]
                full_params[p] = np.arange(low, high + step, step)
                search_params[p] = full_params[p]
            else:
                full_params[p] = np.array([center[p]])

        X = np.array(np.meshgrid(*[full_params[p] for p in self.order])).T.reshape(-1, len(self.order))
        X_search = np.array(np.meshgrid(*[search_params[p] for p in self.space])).T.reshape(-1, len(self.space))
        self.mins = np.min(X_search, axis=0)
        self.maxs = np.max(X_search, axis=0)
        X_norm = 2 * (X_search - self.mins) / (self.maxs - self.mins) - 1
        return X, X_norm

    def sample_initial_points(self, X, X_norm, center, bounds, n_samples, prior):
        """
        Sample initial points from the search space.

        Parameters
        ----------
        X : np.ndarray
            Search space
        X_norm : np.ndarray
            Normalized search space
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter
        n_samples : int
            Number of samples to draw
        prior : bool
            Use prior information for sampling

        Returns
        -------
        np.ndarray
            Sampled points
        """
        if prior:
            means = [center[p] for p in self.space]
            cov = np.diag([((bounds[p][1] - bounds[p][0]) / 5) ** 2 for p in self.space])
            X_free = np.random.multivariate_normal(means, cov, n_samples)
            X_free = np.clip(X_free, self.mins, self.maxs)
            X_norm_samples = 2 * (X_free - self.mins) / (self.maxs - self.mins) - 1
            X_samples = np.tile([center[p] for p in self.order], (n_samples, 1))
            for i, p in enumerate(self.space):
                j = self.order.index(p)
                X_samples[:, j] = X_free[:, i]
            return X_samples, X_norm_samples
        else:
            idx_samples = np.random.choice(X.shape[0], n_samples)
            X_samples = X[idx_samples]
            X_norm_samples = X_norm[idx_samples]
            return X_samples, X_norm_samples

    def score(self, sample, Imin, max_rings, rtol=1e-2):
        """
        Evaluate score at a given sampled geometry.
        
        Parameters
        ----------
        sample : array-like
            Geometry parameters
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance for masking ring pixels

        Returns
        -------
        score : float
            Scalar score for Bayesian optimization
        """
        ttha = calculate_2theta(self.detector, sample)
        min_ttha = np.min(ttha)
        max_ttha = np.max(ttha)
        valid_ttha = self.tth[(self.tth >= min_ttha) & (self.tth <= max_ttha)]

        score = 0.0
        ring = 0
        for tth_i in valid_ttha:
            if ring >= max_rings:
                return score / max_rings
            mask = np.abs(ttha - tth_i) <= rtol * tth_i
            pixels = self.powder[mask]
            pixels = pixels[pixels >= Imin]
            if len(pixels) != 0:
                score += np.sum(pixels)
            ring += 1
        score /= max_rings
        return score

    def pyFAI_score(self, best_param, Imin, max_rings, rtol):
        """
        Evaluate geometry found by BO on pyFAI refinement tool

        Parameters 
        ----------
        best_param : list
            Best parameters found by Bayesian optimization
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance for masking ring pixels

        Returns
        -------
        residual : float
            Residual error after refinement
        score : float
            BO Score of the refined parameters
        params : dict
            Refined parameters
        """
        dist, poni1, poni2, rot1, rot2, rot3 = best_param
        geom_initial = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Best Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_initial,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        self.sg = sg
        score = self.score(best_param, Imin, max_rings, rtol)
        residual = 0
        if len(sg.geometry_refinement.data) > 0:
            residual = sg.geometry_refinement.refine3(fix=["rot3", "wavelength"])
        params = sg.geometry_refinement.param
        return residual, score, params


    @ignore_warnings(category=ConvergenceWarning)
    def sync_bayes_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        rtol,
        beta=1.96,
        prior=True,
        seed=0,
    ):
        """
        Perform Bayesian Optimization on 5 geometric parameters.

        Parameters
        ----------
        center : dict
            Dictionary of the center values for each parameter
        bounds : dict
            Dictionary of the per-parameter bounds for the search space
        res : dict
            Dictionary of the per-parameter resolutions for the search space
        n_samples : int
            Number of initial samples to draw
        n_iterations : int
            Number of optimization iterations
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance in q-space for masking ring pixels
        beta : float
            Exploration-exploitation trade-off parameter for UCB
        prior : bool
            Use prior information for optimization
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed+self.rank)

        # 1. Create Search Space
        X, X_norm = self.create_search_space(bounds, center, res)
        print(f"Rank {self.rank}: Search space size: {X.shape[0]}")

        # 2. Sample Initial Points
        # Rank 0 will sample from a Gaussian prior on center
        # Other ranks will sample uniformly within search space
        if self.rank == 0:
            prior = True
        else:
            prior = False
        X_samples, X_norm_samples = self.sample_initial_points(X, X_norm, center, bounds, n_samples, prior)

        bo_history = {}
        y = np.zeros((n_samples))

        # 3. Evaluate initial points
        for i in range(n_samples):
            y[i] = self.score(X_samples[i], Imin, max_rings, rtol)
            bo_history[f"init_{i+1}"] = {"param": X_samples[i], "score": y[i]}

        tic = time()
        self.comm.Barrier()

        X_samples_all = self.comm.gather(X_samples, root=0)
        X_norm_samples_all = self.comm.gather(X_norm_samples, root=0)
        y_all = self.comm.gather(y, root=0)
        toc = time()

        if self.rank == 0:
            X_samples_all = np.vstack(X_samples_all)
            X_norm_samples_all = np.vstack(X_norm_samples_all)
            y_all = np.concatenate(y_all)
            y_all[np.isnan(y_all)] = 0
            if np.std(y_all) != 0:
                y_norm = (y_all - np.mean(y_all)) / np.std(y_all)
            else:
                y_norm = y_all - np.mean(y_all)

            kernel = RBF(
                length_scale=0.3, length_scale_bounds=(0.2, 0.4)
            ) * ConstantKernel(
                constant_value=1.0, constant_value_bounds=(0.5, 1.5)
            ) + WhiteKernel(
                noise_level=0.001, noise_level_bounds="fixed"
            )
            gp_model = GaussianProcessRegressor(kernel=kernel, random_state=seed)
            gp_model.fit(X_norm_samples_all, y_norm)
            visited_idx = list([])

        for i in range(n_iterations):
            # 4. Rank 0 selects next points with q-UCB
            t0 = time()
            if self.rank == 0:
                nexts = self.q_UCB(X_norm, gp_model, self.size, visited_idx, beta)
                next_points = X[nexts]
                visited_idx.extend(nexts)
            else:
                next_points = None

            # 5. Scatter points to all ranks
            next_point = self.comm.scatter(next_points, root=0)

            # 6. Compute score locally
            score = self.score(next_point, Imin, max_rings, rtol)
            bo_history[f"iter_{i+1}"] = {"param": next_point, "score": score}

            self.comm.Barrier()

            # 7. Gather scores on Rank 0
            score_all = self.comm.gather(score, root=0)

            if self.rank == 0:
                scores = np.array(score_all)
                scores[np.isnan(scores)] = 0
                y_all = np.concatenate([y_all, scores])
                X_samples = np.vstack([X_samples, X[nexts]])
                X_norm_samples = np.vstack([X_norm_samples, X_norm[nexts]])
                if np.std(y_all) != 0:
                    y_norm = (y_all - np.mean(y_all)) / np.std(y_all)
                else:
                    y_norm = y_all - np.mean(y_all)

                # 8. Update Gaussian Process
                gp_model.fit(X_norm_samples, y_norm)
                t1 = time()
                print(f"Iter {i+1}: Total iteration took {t1 - t0:.6f} seconds")

        self.comm.Barrier()

        # 8. Collect BO history from each rank
        bo_histories = self.comm.gather(bo_history, root=0)
        if self.rank == 0:
            for rank in range(self.size):
                history = [bo_histories[r] for r in range(self.size)]

        # 9. Evaluate best geometry using PyFAI refinement tool
        if self.rank == 0:
            best_idx = np.argmax(y_all)
            best_param = X_samples[best_idx]
            residual, score, params = self.pyFAI_score(best_param, Imin, max_rings, rtol)
            result = {
                "history": history,
                "params": params,
                "residual": residual,
                "score": score,
                "best_idx": best_idx,
            }
            return result
        
    @ignore_warnings(category=ConvergenceWarning)
    def async_bayes_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        rtol,
        beta=1.96,
        prior=True,
        seed=0,
    ):
        """
        Perform Bayesian Optimization on 5 geometric parameters.

        Parameters
        ----------
        center : dict
            Dictionary of the center values for each parameter
        bounds : dict
            Dictionary of the per-parameter bounds for the search space
        res : dict
            Dictionary of the per-parameter resolutions for the search space
        n_samples : int
            Number of initial samples to draw
        n_iterations : int
            Number of optimization iterations
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance in q-space for masking ring pixels
        beta : float
            Exploration-exploitation trade-off parameter for UCB
        prior : bool
            Use prior information for optimization
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed+self.rank)

        # 1. Create Search Space
        X, X_norm = self.create_search_space(bounds, center, res)
        print(f"Rank {self.rank}: Search space size: {X.shape[0]}")

        # 2. Sample Initial Points
        # Rank 0 will sample from a Gaussian prior on center
        # Other ranks will sample uniformly within search space
        if self.rank == 0:
            prior = True
        else:
            prior = False
        X_samples, X_norm_samples = self.sample_initial_points(X, X_norm, center, bounds, n_samples, prior)

        bo_history = {}
        y = np.zeros((n_samples))

        # 3. Evaluate initial points
        for i in range(n_samples):
            y[i] = self.score(X_samples[i], Imin, max_rings, rtol)
            bo_history[f"init_{i+1}"] = {"param": X_samples[i], "score": y[i]}

        if np.std(y) != 0:
            y_norm = (y - np.mean(y)) / np.std(y)
        else:
            y_norm = y - np.mean(y)

        kernel = RBF(
            length_scale=0.3, length_scale_bounds=(0.2, 0.4)
        ) * ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + WhiteKernel(
            noise_level=0.001, noise_level_bounds="fixed"
        )
        gp_model = GaussianProcessRegressor(kernel=kernel, random_state=seed)
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        for i in range(n_iterations):
            # 4. Rank 0 selects next points with q-UCB
            t0 = time()
            next = self.UCB(X_norm, gp_model, visited_idx, beta)
            next_point = X[next]
            visited_idx.append(next)

            # 5. Compute score
            score = self.score(next_point, Imin, max_rings, rtol)
            bo_history[f"iter_{i+1}"] = {"param": next_point, "score": score}

            y = np.concatenate([y, [score]])
            X_samples = np.vstack([X_samples, [X[next]]])
            X_norm_samples = np.vstack([X_norm_samples, [X_norm[next]]])
            if np.std(y) != 0:
                y_norm = (y - np.mean(y)) / np.std(y)
            else:
                y_norm = y - np.mean(y)

            # 6. Update Gaussian Process
            gp_model.fit(X_norm_samples, y_norm)
            t1 = time()
            print(f"Iter {i+1}: Total iteration took {t1 - t0:.6f} seconds")

        self.comm.Barrier()

        # 7. Collect BO history from each rank
        bo_histories = self.comm.gather(bo_history, root=0)
        y_all = self.comm.gather(y, root=0)
        X_samples_all = self.comm.gather(X_samples, root=0)
        X_norm_samples_all = self.comm.gather(X_norm_samples, root=0)
        if self.rank == 0:
            for rank in range(self.size):
                history = [bo_histories[r] for r in range(self.size)]
            y = np.concatenate(y_all)
            X_samples = np.vstack(X_samples_all)
            X_norm_samples = np.vstack(X_norm_samples_all)

            # 8. Evaluate best geometry using PyFAI refinement tool
            best_idx = np.argmax(y)
            best_param = X_samples[best_idx]
            residual, score, params = self.pyFAI_score(best_param, Imin, max_rings, rtol)
            result = {
                "history": history,
                "params": params,
                "residual": residual,
                "score": score,
                "best_idx": best_idx,
            }
            return result
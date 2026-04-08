"""2PL IRT utilities aligned with the paper's released implementation."""

import numpy as np
from scipy.special import expit


def logistic(x):
    """Numerically stable sigmoid."""
    return expit(x)


def irt_prob(theta, a, b):
    """
    2PL IRT probability of correct response.

    Args:
        theta: latent ability of the person
        a: item discrimination parameter
        b: item difficulty parameter

    Returns:
        Probability of correct response p(u=1)
    """
    return logistic(a * (theta - b))


def fisher_information(theta, a, b):
    """
    Fisher information for an item at ability theta (Equation 4 in paper).

    I(theta, a, b) = a^2 * logistic(a*(theta-b)) * [1 - logistic(a*(theta-b))]

    Higher Fisher information means the item provides more precise
    ability estimates at that ability level.

    Args:
        theta: current ability estimate
        a: item discrimination
        b: item difficulty

    Returns:
        Fisher information value
    """
    p = logistic(a * (theta - b))
    return a * a * p * (1.0 - p)


def estimate_ability(
    responses,
    a,
    b,
    theta_init=None,
    method="map",
    theta_range=(-4.0, 4.0),
    tol=1e-6,
    max_iter=100,
    prior_mean=0.0,
    prior_std=1.0,
):
    """
    Estimate ability for a 2PL model with MAP or MLE.

    This follows the released FLUID Benchmarking implementation:
    Newton updates with a normal MAP prior and bisection fallback.
    """
    responses = np.asarray(responses, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if responses.shape != a.shape or a.shape != b.shape:
        raise ValueError("responses, a, and b must have the same shape")

    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'")

    mask = ~np.isnan(responses)
    responses = responses[mask]
    a = a[mask]
    b = b[mask]
    if responses.size == 0:
        return float(prior_mean if theta_init is None else theta_init)

    low, high = map(float, theta_range)
    if low >= high:
        raise ValueError("theta_range must satisfy low < high")

    inv_sigma2 = 0.0 if method == "mle" else 1.0 / (prior_std * prior_std)
    theta = prior_mean if theta_init is None else float(theta_init)
    theta = float(np.clip(theta, low, high))

    def score(theta_value):
        p = logistic(a * (theta_value - b))
        prior_term = (prior_mean - theta_value) * inv_sigma2
        likelihood_term = np.sum(a * (responses - p))
        return prior_term + likelihood_term

    def score_prime(theta_value):
        p = logistic(a * (theta_value - b))
        pq = p * (1.0 - p)
        prior_term = -inv_sigma2
        likelihood_term = -np.sum(a * a * pq)
        return prior_term + likelihood_term

    for _ in range(max_iter):
        s = score(theta)
        if abs(s) < tol:
            return float(theta)

        s_prime = score_prime(theta)
        if not np.isfinite(s_prime) or s_prime == 0.0:
            break

        new_theta = theta - (s / s_prime)
        new_theta = float(np.clip(new_theta, low, high))
        current_abs = abs(s)
        for _ in range(15):
            new_score = score(new_theta)
            if abs(new_score) < current_abs or not np.isfinite(new_score):
                break
            new_theta = 0.5 * (new_theta + theta)
        theta = new_theta

    score_low = score(low)
    score_high = score(high)
    if score_low * score_high <= 0:
        lo = low
        hi = high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            score_mid = score(mid)
            if abs(score_mid) < tol:
                return float(mid)
            if score_low * score_mid > 0:
                lo = mid
                score_low = score_mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    return float(high if (score_low > 0 and score_high > 0) else low)


def estimate_ability_se(responses, a, b, theta_hat):
    """
    Estimate standard error of ability estimate.

    SE is approximately 1 / sqrt(sum of Fisher information).
    This is used for dynamic stopping.

    Args:
        responses: array of 0/1 responses
        a: array of item discrimination parameters
        b: array of item difficulty parameters
        theta_hat: estimated ability

    Returns:
        Standard error of ability estimate
    """
    responses = np.array(responses, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    # Sum of Fisher information for all items with responses
    total_info = 0.0
    for i in range(len(responses)):
        info = fisher_information(theta_hat, a[i], b[i])
        total_info += info

    if total_info <= 0:
        return float('inf')
    return 1.0 / np.sqrt(total_info)


class IRTModel:
    """
    Container for pre-fitted IRT model parameters.

    Each benchmark has its own unidimensional IRT model.
    """

    def __init__(self, benchmark, item_ids, a, b):
        """
        Initialize IRT model for a benchmark.

        Args:
            benchmark: benchmark name (e.g., 'arc_challenge')
            item_ids: list of item identifiers
            a: array of discrimination parameters
            b: array of difficulty parameters
        """
        self.benchmark = benchmark
        self.item_ids = np.array(item_ids)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)

        # Create lookup for item parameters
        self._item_params = {
            str(item_id): (float(a_i), float(b_i))
            for item_id, a_i, b_i in zip(item_ids, a, b)
        }

    @classmethod
    def from_csv(cls, benchmark, csv_path):
        """
        Load pre-fitted IRT model from CSV file.

        CSV should have columns: item_id, a, b
        (or first column as item_id, then a, b)
        """
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Handle unnamed first column
        col_names = df.columns.tolist()
        if 'Unnamed: 0' in col_names:
            item_col = 'Unnamed: 0'
        else:
            item_col = col_names[0]

        return cls(
            benchmark=benchmark,
            item_ids=df[item_col].values,
            a=df['a'].values,
            b=df['b'].values
        )

    def get_params(self, item_ids):
        """
        Get IRT parameters for specific items.

        Args:
            item_ids: list of item identifiers

        Returns:
            tuple (a_array, b_array)
        """
        a_list = []
        b_list = []
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self._item_params:
                a_i, b_i = self._item_params[item_id_str]
            elif item_id in self._item_params:
                a_i, b_i = self._item_params[item_id]
            else:
                # Try partial matching for MMLU format
                found = False
                for key in self._item_params:
                    if item_id_str.endswith(key.split('_')[-1]):
                        a_i, b_i = self._item_params[key]
                        found = True
                        break
                if not found:
                    raise ValueError(f"Item {item_id} not found in IRT model")
            a_list.append(a_i)
            b_list.append(b_i)
        return np.array(a_list), np.array(b_list)

    def get_all_params(self):
        """Get parameters for all items."""
        return self.a.copy(), self.b.copy()

    def n_items(self):
        """Number of items in this benchmark."""
        return len(self.item_ids)

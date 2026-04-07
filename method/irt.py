"""
IRT Model: Two-Parameter Logistic (2PL) Item Response Theory model.

Implements the IRT model used in Fluid Benchmarking:
- 2PL model: p(u=1) = logistic(a * (theta - b))
- a = discrimination parameter
- b = difficulty parameter
- theta = latent ability

The paper uses pre-fitted IRT models from the OLL results dataset.
We use those pre-fitted models and implement the ability estimation
and Fisher information functions.
"""

import numpy as np
from scipy.special import expit  # logistic/sigmoid function
from scipy.optimize import minimize_scalar


def logistic(x):
    """Logistic (sigmoid) function."""
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


def estimate_ability(responses, a, b, theta_init=None, method='map'):
    """
    Estimate ability theta for a new person given item responses.

    Uses Maximum A Posteriori (MAP) estimation (Equation 3 in paper).
    Item parameters (a, b) are treated as fixed.

    Args:
        responses: array of 0/1 responses (same length as a, b)
        a: array of item discrimination parameters
        b: array of item difficulty parameters
        theta_init: initial ability estimate (default: 0)
        method: 'map' for MAP estimation (default)

    Returns:
        Estimated ability theta
    """
    responses = np.array(responses, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if theta_init is None:
        theta_init = 0.0

    # Log-likelihood function (negative, to minimize)
    def neg_log_lik(theta):
        p = logistic(a * (theta - b))
        # Clip probabilities to avoid log(0)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
        # MAP with weak prior: N(0, 3) — regularize toward 0
        log_prior = -0.5 * (theta ** 2) / 9.0
        return -(ll + log_prior)

    # Optimize using bounded scalar optimization
    result = minimize_scalar(
        neg_log_lik,
        bounds=(-6, 6),
        method='bounded'
    )
    return float(result.x)


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
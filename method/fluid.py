"""FLUID BENCHMARKING evaluation routines."""

import numpy as np

from .irt import estimate_ability, logistic


def vectorized_fisher_info(theta, a, b):
    """
    Vectorized Fisher information computation.
    Much faster than calling fisher_information in a loop.

    I(theta, a, b) = a^2 * logistic(a*(theta-b)) * [1 - logistic(a*(theta-b))]
    """
    p = logistic(a * (theta - b))
    return a * a * p * (1.0 - p)


def adaptive_select_items(responses, a, b, n_select, theta_init=0.0, return_history=False):
    """
    Select items adaptively using Fisher information (Equation 5 in paper).

    At each step, select the item with highest Fisher information given
    the current ability estimate. Then update the ability estimate with the
    new item's response.

    This is the key algorithmic contribution of FLUID BENCHMARKING.

    Args:
        responses: array of 0/1 responses (length = n_items)
        a: array of discrimination parameters (length = n_items)
        b: array of difficulty parameters (length = n_items)
        n_select: number of items to select
        theta_init: initial ability estimate
        return_history: if True, return ability at each step

    Returns:
        final ability estimate (float)
        (if return_history: list of (n_selected, theta) tuples)
    """
    n_items = len(responses)
    n_select = min(n_select, n_items)

    selected = []
    remaining_mask = np.ones(n_items, dtype=bool)
    theta = theta_init
    history = []

    for _ in range(n_select):
        info = vectorized_fisher_info(theta, a, b)
        info = np.where(remaining_mask, info, -np.inf)
        idx = int(np.argmax(info))
        selected.append(idx)
        remaining_mask[idx] = False

        running_responses = np.full(n_items, np.nan, dtype=float)
        sel = np.array(selected)
        running_responses[sel] = responses[sel]

        theta = estimate_ability(running_responses, a, b, theta_init=theta)
        history.append((len(selected), float(theta), idx))

    if return_history:
        return float(theta), history
    return float(theta)


class FluidBenchmarking:
    """
    FLUID BENCHMARKING evaluator.

    Implements dynamic item selection based on Fisher information
    combined with IRT ability estimation.
    """

    def __init__(self, irt_model, seed=None):
        """
        Initialize FLUID BENCHMARKING with a pre-fitted IRT model.

        Args:
            irt_model: IRTModel instance with item parameters (a, b)
            seed: random seed for reproducibility
        """
        self.irt_model = irt_model
        self.rng = np.random.default_rng(seed)

    def evaluate(self, responses, item_ids=None, n_items=None, theta_init=0.0, return_history=False):
        """
        Evaluate using FLUID BENCHMARKING with dynamic item selection.

        Args:
            responses: list of 0/1 responses (full set of available items)
            item_ids: list of item IDs corresponding to responses
            n_items: number of items to select adaptively (budget)
            theta_init: initial ability estimate (default: 0)
            return_history: if True, return ability estimates at each step

        Returns:
            ability estimate theta
            (if return_history: list of (n_items, theta) tuples)
        """
        responses = np.array(responses, dtype=float)

        if n_items is None:
            n_items = len(responses)

        if item_ids is None:
            a_all, b_all = self.irt_model.get_all_params()
        else:
            a_all, b_all = self.irt_model.get_params(item_ids)

        result = adaptive_select_items(
            responses,
            a_all,
            b_all,
            n_items,
            theta_init=theta_init,
            return_history=return_history,
        )
        return result


def evaluate_with_method(
    irt_model,
    responses,
    item_ids,
    method="fluid",
    n_items=100,
    seed=None,
    theta_init=0.0,
    sample_indices=None,
):
    """
    Evaluate a set of item responses using a specified method.

    Args:
        irt_model: IRTModel instance
        responses: list of 0/1 responses (length = number of available items)
        item_ids: list of item identifiers
        method: 'fluid' (adaptive), 'random_irt' (random + IRT), 'accuracy', 'random'
        n_items: number of items to use (for random selection and fluid budget)
        seed: random seed

    Returns:
        score (ability estimate for IRT methods, accuracy for accuracy method)
    """
    responses = np.array(responses, dtype=float)
    if item_ids is None:
        a_all, b_all = irt_model.get_all_params()
    else:
        a_all, b_all = irt_model.get_params(item_ids)
    n_total = len(a_all)

    if method == 'fluid':
        fb = FluidBenchmarking(irt_model, seed=seed)
        theta = fb.evaluate(
            responses,
            item_ids=item_ids,
            n_items=n_items,
            theta_init=theta_init,
        )
        return theta

    elif method == 'random_irt':
        if sample_indices is None:
            raise ValueError("sample_indices must be provided for random_irt")
        masked_responses = np.full(n_total, np.nan, dtype=float)
        masked_responses[sample_indices] = responses[sample_indices]
        theta = estimate_ability(masked_responses, a_all, b_all, theta_init=theta_init)
        return theta

    elif method == 'accuracy':
        return float(np.mean(responses))

    elif method == 'random':
        if sample_indices is None:
            raise ValueError("sample_indices must be provided for random")
        return float(np.mean(responses[sample_indices]))

    else:
        raise ValueError(f"Unknown method: {method}")

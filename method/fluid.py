"""
Fluid Benchmarking: Dynamic item selection based on Fisher information.

Optimized version with vectorized Fisher information computation.
"""

import numpy as np
from .irt import estimate_ability, fisher_information, logistic


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
    selected_set = set()
    theta = theta_init
    history = [(0, theta)]

    for _ in range(n_select):
        # Compute Fisher information for all remaining items (vectorized)
        info = vectorized_fisher_info(theta, a, b)

        # Mask already selected items
        info_masked = info.copy()
        info_masked[list(selected_set)] = -1

        # Select item with maximum Fisher information
        idx = int(np.argmax(info_masked))
        selected.append(idx)
        selected_set.add(idx)

        # Get responses for selected items
        sel = np.array(selected)
        r_sel = responses[sel]
        a_sel = a[sel]
        b_sel = b[sel]

        # Update ability estimate
        theta = estimate_ability(r_sel, a_sel, b_sel, theta_init=theta)
        history.append((len(selected), theta))

    return theta if not return_history else history


def adaptive_select_items_fast(responses, a, b, n_select, theta_init=0.0):
    """
    Fast adaptive selection using pre-computed Fisher info rankings.

    For large item pools (e.g., MMLU with 14K items), recomputing
    Fisher info at each step is expensive. This version pre-selects
    items based on initial theta and updates theta iteratively.
    """
    n_items = len(responses)
    n_select = min(n_select, n_items)

    # Pre-compute Fisher info at initial theta for all items
    info = vectorized_fisher_info(theta_init, a, b)

    # Get indices of top n_select items by Fisher info
    top_indices = np.argsort(info)[::-1][:n_select]
    selected = list(top_indices)

    # Compute ability estimate with all selected items
    r_sel = responses[selected]
    a_sel = a[selected]
    b_sel = b[selected]
    theta = estimate_ability(r_sel, a_sel, b_sel, theta_init=theta_init)

    # If desired, iteratively refine selection
    # (re-select items at updated theta - only needed for very precise results)
    for _ in range(2):  # 2 refinement iterations
        info = vectorized_fisher_info(theta, a, b)
        top_indices = np.argsort(info)[::-1][:n_select]
        selected = list(top_indices)
        r_sel = responses[selected]
        a_sel = a[selected]
        b_sel = b[selected]
        theta = estimate_ability(r_sel, a_sel, b_sel, theta_init=theta)

    return theta


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

        # Get all item parameters
        a_all, b_all = self.irt_model.get_all_params()

        # Use fast adaptive selection
        theta = adaptive_select_items_fast(
            responses, a_all, b_all, n_items, theta_init=theta_init
        )

        if return_history:
            history = adaptive_select_items(
                responses, a_all, b_all, n_items, theta_init=theta_init,
                return_history=True
            )
            return theta, history

        return theta


def evaluate_with_method(irt_model, responses, item_ids, method='fluid', n_items=100, seed=None):
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
    rng = np.random.default_rng(seed)
    responses = np.array(responses, dtype=float)
    a_all, b_all = irt_model.get_all_params()
    n_total = len(a_all)

    if method == 'fluid':
        fb = FluidBenchmarking(irt_model, seed=seed)
        theta = fb.evaluate(responses, item_ids=item_ids, n_items=n_items, theta_init=0.0)
        return theta

    elif method == 'random_irt':
        # Random selection + IRT ability estimation
        n_use = min(n_items, n_total)
        indices = rng.choice(n_total, size=n_use, replace=False)
        responses_subset = responses[indices]
        a_sub = a_all[indices]
        b_sub = b_all[indices]
        theta = estimate_ability(responses_subset, a_sub, b_sub, theta_init=0.0)
        return theta

    elif method == 'accuracy':
        return float(np.mean(responses))

    elif method == 'random':
        # Random subset accuracy
        n_use = min(n_items, n_total)
        if n_total <= n_use:
            return float(np.mean(responses))
        indices = rng.choice(n_total, size=n_use, replace=False)
        return float(np.mean(responses[indices]))

    else:
        raise ValueError(f"Unknown method: {method}")
"""
Stochastic Beam Search (SBS) implementation for NumPy, mainly based on the implementation in UniqueRandomizer
https://github.com/google-research/unique-randomizer/blob/master/unique_randomizer/stochastic_beam_search.py
with small alterations (e.g., batching SBS to allow higher batch sizes in the policy network).

The implementation is slightly generalized from the description in the paper,
handling the case where not all leaves are at the same level of the tree.
"""
import typing
from typing import Any, Callable, List, Tuple

import numpy as np

State = Any  # Type alias. A state corresponds to a single node (i.e., partial trajectory) in the search tree.
# Leaf node which will be returned by SBS
BeamLeaf = typing.NamedTuple("BeamLeaf", [("state", State),
                                          ("log_probability", float),
                                          ("gumbel", float)
                                          ])


def sample_gumbels_with_maximum(log_probabilities: np.array, target_max: float):
    """Samples a set of gumbels which are conditioned on having a given maximum.
    Based on https://gist.github.com/wouterkool/a3bb2aae8d6a80f985daae95252a8aa8.

    Parameters:
        log_probabilities [np.array]: The log probabilities of the items to sample Gumbels for.
        target_max [float]: The desired maximum sampled Gumbel.

    Returns:
        [np.array] The sampled Gumbels as np.array of same length as `log_probabilities`
    """
    gumbels = np.random.gumbel(loc=log_probabilities)
    max_gumbel = np.max(gumbels)

    # Use equations (23) and (24) in Appendix B.3 of the SBS paper.

    # Note: Numpy may warn "divide by zero encountered in log1p" on the next code
    # line. This is normal and expected, since one element of
    # `gumbels - max_gumbel` should be zero. The math fixes itself later on, and
    # that element ends up being shifted to target_max.
    # We disable this warning locally here.
    with np.errstate(divide='ignore'):
        v = target_max - gumbels + np.log1p(-np.exp(gumbels - max_gumbel))
        ret_gumbels = target_max - np.maximum(v, 0) - np.log1p(np.exp(-np.abs(v)))
    return ret_gumbels


def softmax(x: np.array):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def top_p_filtering(log_probs: np.array, top_p: float):
    sorted_log_probs_indices = np.argsort(- log_probs)  # in descending order
    cumulative_probs = np.cumsum(softmax(log_probs[sorted_log_probs_indices]))

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_log_probs_indices[sorted_indices_to_remove]
    log_probs[indices_to_remove] = np.NINF
    return log_probs


def stochastic_beam_search(
        child_log_probability_fn: Callable[[List[State]], List[np.ndarray]],
        child_transition_fn: Callable[[List[Tuple[State, int]]], List[Tuple[State, bool]]],
        root_states: List[State],
        beam_width: int,
        deterministic: bool = False,
        top_p: float = 0.0
) -> List[List[BeamLeaf]]:
    """Stochastic Beam Search, applied to a batch of states (for higher network throughput).

    Nodes in the beam include "states" which can be anything but must contain
    enough information to:
      1. Define a consistent ordering of all children of the node.
      2. Enumerate the probabilities of all children.
      3. Transition to the state of the child with a given index.

    Parameters:
      child_log_probability_fn: A function that takes a list of states and returns
        the log probabilities of the child states of each input state.

      child_transition_fn: A function that takes a list of (state, i) pairs and maps
        each to a (ith_child, is_leaf) pair. If ith_child is a leaf state, is_leaf
        should be True, and ith_child will potentially be an actual sampled item
        that should be returned by stochastic_beam_search (it may have a different
        form than other non-leaf states).

      root_states: Batch of states of the root node, where the i-th entry corresponds to the root node of the
        i-th problem instance. Root nodes cannot be leaf node.

      beam_width: The desired number of samples, i.e., the width of the beam search.

      deterministic: If True, this falls back to regular beam search.

    Returns:
      A list of up to `beam_width` BeamLeaf objects, corresponding to the sampled leaves.
    """
    k = beam_width
    if beam_width <= 0:
        return []  # trivial case

    batch_len = len(root_states)
    # Storage for nodes which are _on the beam_, that includes leaf states and "internal" states (i.e., states that
    # are not leaves). Within one problem instance, the i-th entries obviously correspond to each other.
    # We just keep the whole tree in a huge array to speed things up.
    batch_leaf_log_probs = [[] for _ in range(batch_len)]
    batch_leaf_gumbels = [[] for _ in range(batch_len)]
    batch_leaf_states = [[] for _ in range(batch_len)]
    batch_internal_log_probs = [[0.0] for _ in range(batch_len)]
    batch_internal_gumbels = [[0.0] for _ in range(batch_len)]
    batch_internal_states = [[root_state] for root_state in root_states]

    # Expand internal nodes until there are none left to expand
    while any([len(internal_states) for internal_states in batch_internal_states]):

        # Batch the internal states and compute the child probabilities in one go for all of them. Note that we obtain
        # them flattened as a list of np.arrays and need to reassign them back to their batches
        flat_child_log_probs_list = child_log_probability_fn([state for internal_states in batch_internal_states for state in internal_states])
        batch_child_log_probs_list = []
        for internal_states in batch_internal_states:
            batch_child_log_probs_list.append(flat_child_log_probs_list[:len(internal_states)])
            flat_child_log_probs_list = flat_child_log_probs_list[len(internal_states):]
        assert len(flat_child_log_probs_list) == 0, "Flat child log probs list should be empty"

        # Make a step of SBS for each item in batch
        for batch_idx in range(batch_len):
            if not len(batch_child_log_probs_list[batch_idx]):
                # Already finished for this problem instance
                continue
            # Expand all feasible internal nodes, however instead of instantiating all states,
            # we simply pack everything into lists and transition only to needed states for efficiency.
            # "All" corresponds to the new children (we will take care of existing leaves later on).
            # All lists below are of same length!
            all_log_probs = []
            all_gumbels = []
            all_states = []  # i-th entry is the parent state of the flattened i-th log_prob/gumbel
            all_child_indices = []

            # Sample Gumbels for children of internal nodes
            for node_state, node_log_prob, node_gumbel, child_log_probs in zip(
                batch_internal_states[batch_idx], batch_internal_log_probs[batch_idx], batch_internal_gumbels[batch_idx],
                batch_child_log_probs_list[batch_idx]
            ):
                if 1 > top_p > 0:
                    # Top-p nucleus sampling and renormalize
                    child_log_probs = top_p_filtering(child_log_probs.copy(), top_p)
                    with np.errstate(divide='ignore'):
                        child_log_probs = np.log(softmax(child_log_probs))
                log_probabilities = child_log_probs + node_log_prob
                # Get only feasible actions
                good_indices = np.where(log_probabilities != np.NINF)[0]
                log_probabilities = log_probabilities[good_indices]
                if deterministic:
                    gumbels = log_probabilities
                else:
                    gumbels = sample_gumbels_with_maximum(log_probabilities, node_gumbel)

                all_log_probs.extend(log_probabilities)
                all_gumbels.extend(gumbels)
                all_states.extend([node_state] * len(log_probabilities))  # repeat parent node for as often as needed
                all_child_indices.extend(good_indices)  # Again, only store feasible actions

            # Now we have all expansions in one list. Select the best k candidates and also take into account
            # current leaf nodes which we might already have (could be that they stay in the beam or fall off).
            leaf_gumbels = batch_leaf_gumbels[batch_idx]
            leaf_log_probs = batch_leaf_log_probs[batch_idx]
            leaf_states = batch_leaf_states[batch_idx]
            num_internal_candidates = len(all_gumbels)
            num_leaf_candidates = len(leaf_gumbels)
            if beam_width >= num_internal_candidates + num_leaf_candidates:
                # All leaf nodes and all internal candidates are selected.
                # So we don't alter the list of leaf nodes.
                to_expand_states = list(zip(all_states, all_child_indices))
                to_expand_log_probs = all_log_probs
                to_expand_gumbels = all_gumbels
            else:
                # Select the unsorted top k in O(num_candidates) time from leaves combined with internal children.
                all_gumbels.extend(leaf_gumbels)
                top_k_indices = np.argpartition(all_gumbels, -k)[-k:]
                to_expand_states = []
                to_expand_log_probs = []
                to_expand_gumbels = []
                leaf_indices = []  # leaf indices which persist
                for i in top_k_indices:
                    if i >= num_internal_candidates:
                        # is a leaf index, so shift index
                        leaf_indices.append(i - num_internal_candidates)
                    else:
                        to_expand_states.append((all_states[i], all_child_indices[i]))
                        to_expand_log_probs.append(all_log_probs[i])
                        to_expand_gumbels.append(all_gumbels[i])

                # Existing leaves in the top k persist
                leaf_log_probs = [leaf_log_probs[i] for i in leaf_indices]
                leaf_gumbels = [leaf_gumbels[i] for i in leaf_indices]
                leaf_states = [leaf_states[i] for i in leaf_indices]

            # Among selected candidates, expand non-leaf nodes
            internal_log_probs = []
            internal_gumbels = []
            internal_states = []
            if len(to_expand_states):
                child_states = child_transition_fn(to_expand_states)
                for log_prob, gumbel, (child_state, is_leaf) in zip(
                    to_expand_log_probs, to_expand_gumbels, child_states
                ):
                    if is_leaf:
                        leaf_log_probs.append(log_prob)
                        leaf_gumbels.append(gumbel)
                        leaf_states.append(child_state)
                    else:
                        internal_log_probs.append(log_prob)
                        internal_gumbels.append(gumbel)
                        internal_states.append(child_state)

            # Put everything back into our batch lists
            batch_leaf_log_probs[batch_idx] = leaf_log_probs
            batch_leaf_gumbels[batch_idx] = leaf_gumbels
            batch_leaf_states[batch_idx] = leaf_states
            batch_internal_log_probs[batch_idx] = internal_log_probs
            batch_internal_gumbels[batch_idx] = internal_gumbels
            batch_internal_states[batch_idx] = internal_states

    # We are done with everything. Pach the leaf data into BeamLeaf objects
    results = []
    for batch_idx in range(batch_len):
        sampled_nodes = []
        for log_prob, gumbel, state in zip(
            batch_leaf_log_probs[batch_idx], batch_leaf_gumbels[batch_idx], batch_leaf_states[batch_idx]
        ):
            sampled_nodes.append(BeamLeaf(state=state, log_probability=log_prob, gumbel=gumbel))

        # Sort the beam in order of decreasing Gumbels. This corresponds to the order
            # one would get by sampling one-at-a-time without replacement.
        results.append(sorted(sampled_nodes, key=lambda x: x.gumbel, reverse=True))

    return results

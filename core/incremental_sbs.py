"""
(Improving) Incremental Stochastic Beam Search (SBS) implementation for NumPy, mainly based on the implementation in UniqueRandomizer
https://github.com/google-research/unique-randomizer/blob/master/unique_randomizer/unique_randomizer.py
with alterations, such as:
    - Our log prob modification, see below.
    - batching SBS to allow higher batch sizes in the policy network
    - flag for keeping memory low (i.e., deleting client states after they have been visited but keeping the logits)

The key idea is to modify the log-probabilities of nodes in sampled trajectories. We do this in two ways:
1.) Subtract the log-prob of a sampled sequence from all intermediate nodes in the sequence. This is incremental
    SBS, where we perform sampling without replacement over many batches.
2.) Use the sampled trajectories to compute the expected objective over the policy (derived from priority sampling),
    then update the log-probs of all intermediate nodes with the advantage of the trajectory over the expectation.
    Note: we need to be careful about proper normalization.
"""
import typing

import sys
import numpy as np
import core.stochastic_beam_search as sbs
from typing import Callable, List, Optional, Tuple

sys.setrecursionlimit(10000)  # Policy updates are performed recursively right now. Could be improved in the future.


def log_subtract(x: float, y: float) -> float:
    """Returns log(exp(x) - exp(y)), or negative infinity if x <= y."""
    # Inspired by https://stackoverflow.com/questions/778047.
    with np.errstate(divide='ignore'):
        result = x + np.log1p(-np.exp(np.minimum(y - x, 0)))
    return result


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel.
    Adapted from `https://github.com/wouterkool/stochastic-beam-search/blob/stochastic-beam-search/fairseq/gumbel.py`.
    In practice, we will need to set x := \kappa - \phi_i
    """
    with np.errstate(divide='ignore'):
        y = np.exp(-x)
        result = np.where(
            x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
            -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
            np.log(-np.expm1(-np.exp(-x)))  # Hope for the best
        )
    return result


def gumbel_without_replacement_expectation(
        sbs_leaves: List[sbs.BeamLeaf],
        leaf_eval_fn: Callable,
        normalize_importance_weights: bool = True) -> float:
    """
    Compute the expected outcome of rolling out the current policy given the sampled trajectories from SBS.
    For details of how to compute it, we refer to the original SBS paper
    "https://proceedings.mlr.press/v97/kool19a/kool19a.pdf", Section 4.2 "BlEU score estimation".

    Parameters:
        sbs_leaves [List[sbs.BeamLeaf]]: Sampled trajectories from which we compute the expected value.
        leaf_eval_fn: [Callable]
        normalize_importance_weights [bool]: In practice, the importance weighted estimator can have high variance,
            and it can be preferable to normalize them by their sum. Default is `True`.

    Returns:
        expected_outcome [float]: Expectation computed from sbs_leaves using optional importance weights.
    """
    if len(sbs_leaves) == 1:
        return leaf_eval_fn(sbs_leaves[0].state[1])
    # Sort leaves in descending order by their sampled Gumbels
    sbs_leaves = sorted(sbs_leaves, key=lambda x: x.gumbel, reverse=True)

    # We need the log-probs, gumbels and outcomes for each leaf
    log_probs = np.array([leaf.log_probability for leaf in sbs_leaves])
    outcomes = np.array([leaf_eval_fn(leaf.state[1]) for leaf in sbs_leaves])

    # We have the beam leaves sorted by their sampled Gumbels, so the last element is the smallest gumbel
    kappa = sbs_leaves[-1].gumbel
    # Discard the last entry, we won't need it
    log_probs = log_probs[:-1]
    _outcomes = outcomes[:-1]
    # See Appendix C "Numerical stability of importance weights" in SBS paper
    importance_weights = np.exp(log_probs - gumbel_log_survival(kappa - log_probs))
    if normalize_importance_weights:
        importance_weights = importance_weights / np.sum(importance_weights)
    expected_outcome = float(np.sum(importance_weights * _outcomes))
    return expected_outcome


class _TrieNode(object):
    """A trie node as in UniqueRandomizer.

    Attributes:
        parent: The _TrieNode parent of this node, or `None` if this node is the root.
        index_in_parent: The action index of this node in the parent, or `None` if this node
            is the root.
        children: A list of _TrieNode children. A child may be `None` if it is not
            expanded yet. The entire list will be `None` if this node has never visited
            a child yet. The list will be empty if this node is a leaf in the trie.
        unsampled_log_masses: A numpy array containing the current (!!) (unsampled) log
            probability mass of each child, or `None` if this node has never sampled a
            child yet.
        children_advantages: A numpy array containing the cumulated advantage of the node's children.
        sbs_child_state_cache: Used for caching children's states when performing SBS incrementally.
    """

    def __init__(self, parent: Optional['_TrieNode'], index_in_parent: Optional[int]) -> None:
        """Initializes a _TrieNode.

        Parameters:
            parent [Optional[_TrieNode]]: The parent of this node, or `None` if this node is the root.
            index_in_parent [Optional[int]]: This node's action index in the parent node, or `None` if this
                node is the root.
        """
        self.parent = parent
        self.index_in_parent = index_in_parent
        self.children = None
        self.unsampled_log_masses = None
        self.children_advantages = None
        # Counts the number of times we have traversed this node
        self.children_visit_counts = None
        # Caches (state, is_leaf)-tuples obtained in SBS so we only transition to new states if needed.
        self.sbs_child_state_cache: Optional[List[Tuple[sbs.State, bool]]] = None
        if self.parent is None:
            self.ancestors = []  # List of ancestor nodes
        else:
            self.ancestors = [self.parent] + self.parent.ancestors

        # Flag to track in a given round if the advantage of the node was already calculated.
        self.advantage_touched_flag = False

    def initial_log_mass_if_not_sampled(self) -> float:
        """Returns this node's initial log probability mass.

        This assumes that no samples have been drawn from this node yet.
        """
        # If no samples have been drawn yet, the unsampled log mass equals the
        # desired initial log mass.
        return (self.parent.unsampled_log_masses[self.index_in_parent]
                # If the node is the root, the initial log mass is 0.0.
                if self.parent else 0.0)

    def mark_leaf(self) -> None:
        """Marks this node as a leaf."""
        self.children = []

    def exhausted(self) -> bool:
        """Returns whether all of the mass at this node has been sampled."""
        # Distinguish [] and None.
        if self.children is not None and not len(self.children):
            return True
        if self.unsampled_log_masses is None:
            return False  # This node is not a leaf but has never been sampled from.
        return all(np.isneginf(self.unsampled_log_masses))

    def mark_mass_sampled(self, log_mass: float) -> None:
        """Recursively subtracts log_mass from this node and its ancestors."""
        if not self.parent:  # is the root.
            return
        if self.exhausted():
            new_log_mass = -np.inf  # explicitly set the node's log_mass to -inf to prevent sampling from it again.
        else:
            new_log_mass = log_subtract(self.parent.unsampled_log_masses[self.index_in_parent], log_mass)
        self.parent.unsampled_log_masses[self.index_in_parent] = new_log_mass
        self.parent.mark_mass_sampled(log_mass)

    def add_simple_advantage(self, mass: float) -> None:
        """Recursively adds mass to this node and its ancestors."""
        if not self.parent:  # is the root
            return
        self.parent.children_advantages[self.index_in_parent] = self.parent.children_advantages[self.index_in_parent] + mass
        self.parent.children_visit_counts[self.index_in_parent] = self.parent.children_visit_counts[self.index_in_parent] + 1
        self.parent.add_simple_advantage(mass)

    def obtain_locally_estimated_advantage(self, root_expectation: float, advantage_constant: float, beam_leaves: List, leaf_eval_fn: Callable):
        if not self.parent or self.advantage_touched_flag:  # is the root or it was already updated (and thus, also its ancestors)
            return
        # Check if this is a leaf node. If so, we don't need to update the policy here because we already mark it
        # as sampled anyway, so we just continue in this case.
        if len(self.children):
            self.advantage_touched_flag = True
            # Get all beam_leaves which share the parent node of this one
            shared_parent_leaves = []
            for beam_leaf in beam_leaves:
                if self.parent in beam_leaf.state[0].ancestors:
                    shared_parent_leaves.append(beam_leaf)

            # If we have at least two trajectories which run through the parent node (including the current),
            # we estimate the expectation of the parent node.

            # Compute the expected value of the parent node.
            if len(shared_parent_leaves) >= 2:
                expectation_parent = gumbel_without_replacement_expectation(shared_parent_leaves, leaf_eval_fn)

                shared_node_leaves = []
                for beam_leaf in shared_parent_leaves:
                    if self in beam_leaf.state[0].ancestors:
                        shared_node_leaves.append(beam_leaf)

                expectation_node = gumbel_without_replacement_expectation(shared_node_leaves, leaf_eval_fn)
                # Compute the advantage and add it
                self.parent.children_advantages[self.index_in_parent] = self.parent.children_advantages[self.index_in_parent] + advantage_constant * (expectation_node - expectation_parent)

        self.parent.obtain_locally_estimated_advantage(root_expectation, advantage_constant, beam_leaves, leaf_eval_fn)


class IncrementalSBS:
    """
    Main class for incrementally performing Stochastic Beam Search and updating the logits of subsequences met on the way.

    Construct an instance of this class for a batch of "initial states" corresponding to problem instances.
    """
    def __init__(self, initial_states: List[sbs.State],
                 child_log_probability_fn: Callable[[List[sbs.State]], List[np.ndarray]],
                 child_transition_fn: Callable[[List[Tuple[sbs.State, int]]], List[Tuple[sbs.State, bool]]],
                 leaf_evaluation_fn: Optional[Callable[[sbs.State], float]] = None,
                 memory_aggressive: bool = False):
        """
        Parameters:
            initial_states: List of initial states used as root nodes.

            child_log_probability_fn: A function that takes a list of states and returns
                the log probabilities of the child states of each input state.

            child_transition_fn: A function that takes a list of (state, i) pairs and maps
                each to a (ith_child, is_leaf) pair. If ith_child is a leaf state, is_leaf
                should be True, and ith_child will potentially be an actual sampled item
                that should be returned by stochastic_beam_search (it may have a different
                form than other non-leaf states).
                (Wrapped and passed directly to sbs.stochastic_beam_search)

            leaf_evaluation_fn: An optional function that takes the sbs.State of a leaf (i.e., a finished trajectory)
                and returns some "outcome" of the trajectory, such as an objective. Must be provided for
                "policy_improvement"-type updates where the log-probs are updated according to their advantage,
                so we update with them with the goal of MAXIMIZING the outcome. If the original goal of
                the problem is to minimize some objective (e.g., routing problems), make sure that you flip the
                sign of the objective.

            memory_aggressive [bool]: If this is True, the internal states in the search tree are erased after passing them,
                thus on the one hand needing to re-transition to them when passing them again, but on the other hand
                saving memory.
        """
        # A node will always be a tuple (_TrieNode, sbs.State).
        self.root_nodes = [
            (_TrieNode(None, None), state)
            for state in initial_states
        ]
        self.root_nodes_exhausted = [False] * len(self.root_nodes)  # will be True if all probability mass has been returned
        self.leaf_evaluation_fn = leaf_evaluation_fn
        self.child_log_probability_fn = child_log_probability_fn
        self.child_transition_fn = child_transition_fn
        self.memory_aggressive = memory_aggressive

    def perform_incremental_sbs(self, beam_width: int, num_rounds: int, log_prob_update_type: str = "reduce_mass", advantage_constant: float = 1.,
                                min_max_normalize_advantage: bool = False,
                                expected_value_use_simple_mean: bool = False,
                                use_pure_outcomes: bool = False,
                                normalize_advantage_by_visit_count: bool = False,
                                perform_first_round_deterministic: bool = False,
                                min_nucleus_top_p: float = 1.
                                ) -> List[List[sbs.BeamLeaf]]:
        """
        Performs incremental SBS with the given type of updating the log-probs. Note that the trie and all log-prob updates
        persist, so calling the method multiple times will not reset the trie.

        Parameters:
            beam_width [int]: Beam width for one round of SBS
            num_rounds [int]: Number of SBS rounds, where we update the log-probs after each round.
            log_prob_update_type [str]: Must be "wor" (sample without replacement), "gumbeldore" (our method),
                or "theory_gumbeldore" (locally estimated advantage with theoretical policy improvement)
            advantage_constant [float]: "Sigma" in the paper. For policy_improvement, we update the logits by adding the normalized advantage
                times `advantage_constant`. This should be tuned manually specifically to the problem at hand.
            min_max_normalize_advantage [bool]: If True, we perform a min max normalization on the advantages so that
                the highest/lowest advantage is 1/-1.
            expected_value_use_simple_mean [bool]: If True, we do not compute the expected value using Gumbels and
                importance sampling, but simply calculate the arithmetic mean of the leaf outcomes.
            use_pure_outcomes [bool]: If True, we do not subtract the expected value from the actual outcomes, but
                simply return the outcome uncentered. This can be desirable in, e.g., a board game situation, where
                we want to value wins/losses regardless of the expected value of the trajectories. In particular,
                we use it in Gomoku.
            normalize_advantage_by_visit_count [bool]: If True, we divide the accumulated advantage of a node by its
                visit count. This shifts the search method further towards a Gumbel AlphaZero-type update. Can be beneficial
                in conjunction with `use_pure_outcomes` set to True.
            perform_first_round_deterministic [bool]: If True, the first round is a simple beam search round,
                to get the exploitation of the policy "out of the system". Their mass will be reduced, but no advantage
                is propagated.
            min_nucleus_top_p [float]: If smaller than 1, couples SBS with Top-p (nucleus) sampling, where we
                increase p linearly to 1 starting from `min_nucleus_top_p` over the rounds.
        """
        leaves_batch: List[List[sbs.BeamLeaf]] = [[] for _ in range(len(self.root_nodes))]

        for round_idx in range(num_rounds):
            # Check for each root node if its exhausted. If so, we won't search it again
            unexhausted_root_idcs = [i for i, exhausted in enumerate(self.root_nodes_exhausted) if not exhausted]
            root_nodes_to_search = [self.root_nodes[i] for i in unexhausted_root_idcs]

            is_deterministic_round = True if perform_first_round_deterministic and round_idx == 0 else False

            top_p_round = 1. if num_rounds == 1 else (1 - round_idx / (num_rounds - 1.)) * min_nucleus_top_p + 1. * (round_idx / (num_rounds - 1.))
            round_beam_leaves_batch = sbs.stochastic_beam_search(
                child_log_probability_fn=self.wrap_child_log_probability_fn(
                    self.child_log_probability_fn,
                    normalize_advantage_by_visit_count
                ),
                child_transition_fn=self.wrap_child_transition_fn(self.child_transition_fn, self.memory_aggressive),
                root_states=root_nodes_to_search,
                beam_width=beam_width,
                deterministic=is_deterministic_round,
                top_p=1 if is_deterministic_round else top_p_round
            )

            # Update probabilities and remove _TrieNode parts of the leaves.
            for j, beam_leaves in enumerate(round_beam_leaves_batch):
                batch_idx = unexhausted_root_idcs[j]
                # The first thing we do is that we compute the expected value of the outcome over the current policy
                if not is_deterministic_round and log_prob_update_type == "gumbeldore":
                    expected_outcome, outcomes, normalized_advantages = self.compute_expected_outcome(
                        sbs_leaves=beam_leaves,
                        simple_mean=expected_value_use_simple_mean,
                        min_max_normalize_advantage=min_max_normalize_advantage,
                        use_pure_outcomes=use_pure_outcomes
                    )
                # here we replace the beam leaf state consisting of (_TrieNode, sbs.State) tuples
                # to only contain the sbs.State
                client_beam_leaves = []
                for i, beam_leaf in enumerate(beam_leaves):
                    leaf_node, client_state = beam_leaf.state

                    if log_prob_update_type in ["wor", "gumbeldore", "theory_gumbeldore"]:
                        log_sampled_mass = leaf_node.initial_log_mass_if_not_sampled()
                        leaf_node.mark_mass_sampled(log_sampled_mass)
                    if not is_deterministic_round:
                        if log_prob_update_type == "gumbeldore":
                            # Compute \sigma(x) = c * x
                            leaf_node.add_simple_advantage(advantage_constant * normalized_advantages[i])
                        elif log_prob_update_type == "theory_gumbeldore":
                            root_expectation = gumbel_without_replacement_expectation(beam_leaves, self.leaf_evaluation_fn)
                            leaf_node.obtain_locally_estimated_advantage(root_expectation, advantage_constant, beam_leaves, self.leaf_evaluation_fn)

                    client_beam_leaves.append(beam_leaf._replace(state=client_state))
                leaves_batch[batch_idx].extend(client_beam_leaves)
                self.root_nodes_exhausted[batch_idx] = self.root_nodes[batch_idx][0].exhausted()

        return leaves_batch

    def compute_expected_outcome(self,
                                 sbs_leaves: List[sbs.BeamLeaf],
                                 simple_mean: bool = False,
                                 normalize_importance_weights: bool = True,
                                 min_max_normalize_advantage: bool = False,
                                 use_pure_outcomes: bool = False) \
            -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute the expected outcome of rolling out the current policy given the sampled trajectories from SBS.
        For details of how to compute it, we refer to the original SBS paper
        "https://proceedings.mlr.press/v97/kool19a/kool19a.pdf", Section 4.2 "BlEU score estimation".

        Parameters:
            sbs_leaves [List[sbs.BeamLeaf]]: Sampled trajectories from which we compute the expected value.
            simple_mean [bool]: If True, we do not compute the expected outcome using the Gumbels, but merely perform
                a simple average over the leaf outcomes.
            normalize_importance_weights [bool]: In practice, the importance weighted estimator can have high variance,
                and it can be preferable to normalize them by their sum. Default is `True`.
            min_max_normalize_advantage [bool]: If True, we perform a min max normalization on the advantages so that
                the highest/lowest advantage is 1/-1.
            use_pure_outcomes [bool]: If True, we do not subtract the expected value from the actual outcomes, but
                simply return the outcome uncentered. This can be desirable in, e.g., a board game situation, where
                we want to value wins/losses regardless of the expected value of the trajectories.

        Returns:
            expected_outcome [float]: Expectation computed from sbs_leaves using importance weights.
            outcomes [np.array]: Full array of outcomes of leaves.
            normalized_advantages [np.array]: Advantages normalized such that the maximum positive in the beam
                has advantage 1, the expected value has advantage 0 and the minimum negative has advantage -1.
        """
        # We need the log-probs, gumbels and outcomes for each leaf
        log_probs = np.array([leaf.log_probability for leaf in sbs_leaves])
        outcomes = np.array([self.leaf_evaluation_fn(leaf.state[1]) for leaf in sbs_leaves])
        min_outcome = np.min(outcomes)
        max_outcome = np.max(outcomes)
        if simple_mean:
            expected_outcome = float(np.mean(outcomes))
        else:
            # SBS returns the beam leaves sorted by their sampled Gumbels, so the last element is the smallest gumbel
            kappa = sbs_leaves[-1].gumbel
            # Discard the last entry, we won't need it
            log_probs = log_probs[:-1]
            _outcomes = outcomes[:-1]
            # See Appendix C "Numerical stability of importance weights" in SBS paper
            importance_weights = np.exp(log_probs - gumbel_log_survival(kappa - log_probs))
            if normalize_importance_weights:
                importance_weights = importance_weights / np.sum(importance_weights)
            expected_outcome = float(np.sum(importance_weights * _outcomes))

        if not use_pure_outcomes:
            advantages = outcomes - expected_outcome
            if min_max_normalize_advantage:
                advantages = np.where(
                    advantages > 0,
                    advantages / (np.abs(max_outcome - expected_outcome) + 1e-7),
                    advantages / (np.abs(min_outcome - expected_outcome) + 1e-7),
                )
        else:
            advantages = outcomes

        return expected_outcome, outcomes, advantages

    @staticmethod
    def wrap_child_log_probability_fn(child_log_probability_fn, normalize_advantage_by_visit_count: bool) -> Callable[[List[Tuple[_TrieNode, sbs.State]]], List[np.ndarray]]:
        def wrapper_child_log_probability_fn(node_state_tuples: List[Tuple[_TrieNode, sbs.State]]) -> List[np.ndarray]:
            """Computes child probabilities while updating the trie."""
            results = [None] * len(node_state_tuples)
            unexpanded_client_states = []  # States for which we haven't computed log probs yet
            unexpanded_indices = []  # Corresponding to index in `node_state_tuples`

            for i, (node, client_state) in enumerate(node_state_tuples):
                if node.unsampled_log_masses is None:
                    # We haven't computed log probs for this node yet.
                    unexpanded_client_states.append(client_state)
                    unexpanded_indices.append(i)
                else:
                    # We already have computed log probabilities for this node (and also may have already updated them!)
                    # However, the log probs might not be normalized, so we need to normalize them.
                    # Note that normalizing the log probs is the same as if we would first subtract the parent's log mass to obtain
                    # the conditional log probs and then normalize them again
                    advantages = node.children_advantages
                    if normalize_advantage_by_visit_count:
                        advantages = node.children_advantages / node.children_visit_counts

                    # Add advantage
                    log_unnormalized = node.unsampled_log_masses + advantages

                    unnormalized = np.exp(log_unnormalized - np.max(log_unnormalized))
                    with np.errstate(divide='ignore'):
                        results[i] = np.log(unnormalized / np.sum(unnormalized))

            # Use client's child_log_probability_fn to get probabilities for unexpanded states.
            if unexpanded_client_states:
                client_fn_results = child_log_probability_fn(unexpanded_client_states)
                for i, log_probs in zip(unexpanded_indices, client_fn_results):
                    results[i] = log_probs
                    # also set the log probs on the node for which we computed them
                    node = node_state_tuples[i][0]
                    node.unsampled_log_masses = log_probs + node.initial_log_mass_if_not_sampled()
                    node.children_advantages = np.zeros(len(node.unsampled_log_masses))
                    node.children_visit_counts = np.full(len(node.unsampled_log_masses), 1e-8)

            return typing.cast(List[np.ndarray], results)

        return wrapper_child_log_probability_fn

    @staticmethod
    def wrap_child_transition_fn(child_transition_fn, memory_aggressive: bool) -> Callable[[List[Tuple[Tuple[_TrieNode, sbs.State], int]]],
                                                                  List[Tuple[Tuple[_TrieNode, sbs.State], bool]]]:
        def wrapper_child_transition_fn(node_state_action_index_pairs: List[Tuple[Tuple[_TrieNode, sbs.State], int]]) -> List[Tuple[Tuple[_TrieNode, sbs.State], bool]]:
            """Computes child states while updating the trie."""
            results = [None] * len(node_state_action_index_pairs)
            unexpanded_client_state_index_pairs = []  # States for which we haven't computed the transition yet
            unexpanded_indices = []  # Corresponding to index in `node_state_action_index_pairs`

            for i, ((node, client_state), child_index) in enumerate(node_state_action_index_pairs):
                # Initialize children structures if needed. We can be sure that `unsampled_log_masses` is not None
                if node.children is None:
                    num_children = len(node.unsampled_log_masses)
                    node.children = [None] * num_children
                    node.sbs_child_state_cache = [None] * num_children

                if node.children[child_index] is None or node.sbs_child_state_cache[child_index] is None:
                    # The child has not been created before, or there is no entry for the child in the sbs cache.
                    # The latter only happens in memory_aggressive mode
                    unexpanded_client_state_index_pairs.append((client_state, child_index))
                    unexpanded_indices.append(i)
                else:
                    # There already is a child which we can use.
                    child_client_state, child_is_leaf = node.sbs_child_state_cache[child_index]
                    node.children[child_index].advantage_touched_flag = False
                    results[i] = ((node.children[child_index], child_client_state), child_is_leaf)

            # Use client's child_transition_fn to get child client states
            if unexpanded_client_state_index_pairs:
                client_fn_results = child_transition_fn(unexpanded_client_state_index_pairs)
                for i, (child_client_state, child_is_leaf) in zip(unexpanded_indices, client_fn_results):
                    (node, _), child_index = node_state_action_index_pairs[i]
                    child_node = node.children[child_index]
                    if node.children[child_index] is None:
                        # Usual case: the node has not been created before
                        # This condition is only unmet in memory aggressive case where the child node exists but
                        # the client state is no longer there.
                        child_node = _TrieNode(parent=node, index_in_parent=child_index)
                        if child_is_leaf:
                            child_node.mark_leaf()
                        node.children[child_index] = child_node

                    node.sbs_child_state_cache[child_index] = (child_client_state, child_is_leaf)
                    results[i] = ((child_node, child_client_state), child_is_leaf)

            if memory_aggressive:
                # for each node from which we have transitioned, remove its client state from the parent
                for (node, _), _ in node_state_action_index_pairs:
                    if node.parent is not None:
                        node.parent.sbs_child_state_cache[node.index_in_parent] = None

            return typing.cast(List[Tuple[Tuple[_TrieNode, sbs.State], bool]], results)

        return wrapper_child_transition_fn

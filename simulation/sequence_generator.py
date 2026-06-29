"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequence_generator.py
@Time    : 09/01/2026 11:20
@Desc    :
Sequence generation module for sensitivity analysis.

This module provides functions to generate sequences in spell-based representation
(DSS + durations) and convert them to position-level sequences. It supports
various generation strategies needed for different simulation strands.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class SequenceGenerator:
    """
    Generator for spell-based sequences.
    
    Sequences are represented as:
    - DSS (Distinct Successive States): list of state labels, e.g., ['a', 'b', 'c']
    - Durations: list of spell lengths summing to total_length, e.g., [5, 8, 7]
    
    This representation can be expanded to position-level sequences where needed.
    """
    
    def __init__(self, state_labels: List[str], total_length: int, random_seed: int = 42):
        """
        Initialize sequence generator.
        
        Parameters
        ----------
        state_labels : List[str]
            List of state labels (e.g., ['a', 'b', 'c', 'd', 'e'])
        total_length : int
            Fixed total length T for all sequences
        random_seed : int
            Random seed for reproducibility
        """
        self.state_labels = state_labels
        self.n_states = len(state_labels)
        self.total_length = total_length
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _validate_integer_durations(durations: np.ndarray) -> np.ndarray:
        raw = np.asarray(durations)

        if not np.issubdtype(raw.dtype, np.number):
            raise ValueError("Spell durations must be numeric.")
        if not np.all(np.isfinite(raw)):
            raise ValueError("Spell durations must be finite.")
        if not np.all(raw == np.floor(raw)):
            raise ValueError("Spell durations must be integers.")

        return raw.astype(int)
    
    def generate_random_durations(self, n_spells: int, total_length: Optional[int] = None) -> np.ndarray:
        """
        Generate random spell durations that sum to total_length.
        
        Uses a uniform distribution over compositions (ordered partitions into
        positive integers). Each duration must be at least 1 to ensure valid spells.
        The method selects n_spells-1 cut points uniformly from valid positions.
        
        Parameters
        ----------
        n_spells : int
            Number of spells (length of DSS)
        total_length : int, optional
            Target sum for durations. If None, uses self.total_length.
        
        Returns
        -------
        np.ndarray
            Array of durations summing to total_length, each >= 1
        """
        if total_length is None:
            total_length = self.total_length

        if n_spells < 1:
            raise ValueError("n_spells must be >= 1.")
        if total_length < 1:
            raise ValueError("total_length must be >= 1.")
        
        if n_spells > total_length:
            raise ValueError(f"Cannot have {n_spells} spells when total_length={total_length}")
        
        if n_spells == 1:
            return np.array([total_length])
        
        # Generate n_spells-1 random cut points in [1, total_length-1]
        # Then compute durations as differences between consecutive cut points
        cut_points = sorted(self.rng.choice(
            range(1, total_length), 
            size=n_spells - 1, 
            replace=False
        ))
        cut_points = [0] + list(cut_points) + [total_length]
        
        durations = np.diff(cut_points).astype(int)
        return durations
    
    def generate_random_dss(self, n_spells: int, allow_repeats: bool = True, no_adjacent_repeats: bool = True) -> List[str]:
        """
        Generate random DSS (Distinct Successive States).
        
        Parameters
        ----------
        n_spells : int
            Number of spells (length of DSS)
        allow_repeats : bool
            Whether to allow repeated states in DSS (default: True)
            If False, ensures all states are distinct
        no_adjacent_repeats : bool
            Whether to disallow adjacent repeated states (default: True)
            This ensures swap perturbations are always effective
        
        Returns
        -------
        List[str]
            List of state labels
        
        Raises
        ------
        ValueError
            If constraints cannot be satisfied (e.g., no_adjacent_repeats=True
            with only 1 state and n_spells > 1)
        """
        if n_spells < 1:
            raise ValueError("n_spells must be >= 1.")
        if self.n_states < 1:
            raise ValueError("state_labels must contain at least one state.")

        if not allow_repeats and n_spells > self.n_states:
            raise ValueError(
                f"Cannot generate {n_spells} distinct spells with only {self.n_states} states"
            )
        
        if allow_repeats and no_adjacent_repeats:
            # Check if constraint can be satisfied
            if self.n_states == 1 and n_spells > 1:
                raise ValueError(
                    f"Cannot satisfy no_adjacent_repeats=True with only 1 state "
                    f"and n_spells={n_spells} > 1"
                )
            
            # Generate DSS ensuring no adjacent repeats
            dss = []
            for i in range(n_spells):
                if i == 0:
                    # First spell: any state
                    dss.append(self.rng.choice(self.state_labels))
                else:
                    # Subsequent spells: avoid repeating previous state
                    available = [s for s in self.state_labels if s != dss[-1]]
                    dss.append(self.rng.choice(available))
            return dss
        elif allow_repeats:
            return list(self.rng.choice(self.state_labels, size=n_spells))
        else:
            return list(self.rng.choice(self.state_labels, size=n_spells, replace=False))
    
    def generate_sequence(
        self, 
        dss: Optional[List[str]] = None,
        durations: Optional[np.ndarray] = None,
        n_spells: Optional[int] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generate a single sequence in spell representation.
        
        Parameters
        ----------
        dss : List[str], optional
            Pre-specified DSS. If None, generates randomly.
        durations : np.ndarray, optional
            Pre-specified durations. If None, generates randomly.
        n_spells : int, optional
            Number of spells. Required if both dss and durations are None.
        
        Returns
        -------
        Tuple[List[str], np.ndarray]
            (DSS, durations) tuple
        
        Raises
        ------
        ValueError
            If dss and durations have inconsistent lengths, or if durations
            contain invalid values (non-positive or wrong sum), or if dss
            contains invalid state labels.
        """
        # Determine number of spells
        if dss is not None:
            n_spells = len(dss)
        elif durations is not None:
            n_spells = len(durations)
        elif n_spells is None:
            raise ValueError("Must specify either dss, durations, or n_spells")
        
        # Generate DSS if not provided
        if dss is None:
            dss = self.generate_random_dss(n_spells)
        if any(a == b for a, b in zip(dss, dss[1:])):
            raise ValueError(f"DSS contains adjacent duplicate states: {dss}")
        
        # Generate durations if not provided
        if durations is None:
            durations = self.generate_random_durations(n_spells)
        else:
            durations = self._validate_integer_durations(durations)
        
        # Validate consistency
        if len(dss) != len(durations):
            raise ValueError(
                f"DSS length {len(dss)} != durations length {len(durations)}"
            )
        
        # Validate state labels
        invalid_labels = [s for s in dss if s not in self.state_labels]
        if invalid_labels:
            raise ValueError(
                f"Invalid state labels in DSS: {invalid_labels}. "
                f"Valid labels: {self.state_labels}"
            )
        
        # Validate durations
        if np.any(durations <= 0):
            raise ValueError(
                f"All durations must be >= 1, got: {durations}"
            )
        
        # Validate that durations sum to total_length
        if durations.sum() != self.total_length:
            raise ValueError(
                f"Durations sum to {durations.sum()}, expected {self.total_length}"
            )
        
        return (dss, durations)
    
    def expand_to_position_sequence(
        self, 
        dss: List[str], 
        durations: np.ndarray
    ) -> List[str]:
        """
        Expand spell representation to position-level sequence.
        
        Converts (DSS, durations) to a sequence of length total_length where
        each position contains the state label for that time point.
        
        Example:
            dss = ['a', 'b', 'c']
            durations = [5, 8, 7]
            -> ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 
                'c', 'c', 'c', 'c', 'c', 'c', 'c']
        
        Parameters
        ----------
        dss : List[str]
            Distinct Successive States
        durations : np.ndarray
            Spell durations
        
        Returns
        -------
        List[str]
            Position-level sequence of length total_length
        
        Raises
        ------
        ValueError
            If the expanded sequence length does not match total_length
        """
        if len(dss) != len(durations):
            raise ValueError(
                f"DSS length {len(dss)} != durations length {len(durations)}."
            )
        durations = self._validate_integer_durations(durations)
        if np.any(durations <= 0):
            raise ValueError("All spell durations must be positive.")
        sequence = []
        for state, duration in zip(dss, durations):
            sequence.extend([state] * duration)
        
        if len(sequence) != self.total_length:
            raise ValueError(
                f"Expanded sequence length {len(sequence)} != total_length {self.total_length}. "
                f"Durations sum to {durations.sum()}"
            )
        
        return sequence
    
    def expand_partial(
        self,
        dss: List[str],
        durations: np.ndarray
    ) -> List[str]:
        """
        Expand spell representation to position-level sequence without total_length check.
        Use for partial sequences (e.g. suffix) where durations sum to a length < total_length.
        """
        if len(dss) != len(durations):
            raise ValueError(
                f"DSS length {len(dss)} != durations length {len(durations)}."
            )
        durations = self._validate_integer_durations(durations)
        if np.any(durations <= 0):
            raise ValueError("All spell durations must be positive.")
        sequence = []
        for state, duration in zip(dss, durations):
            sequence.extend([state] * duration)
        return sequence
    
    def generate_group(
        self,
        n_sequences: int,
        dss_template: Optional[List[str]] = None,
        duration_constraints: Optional[Dict[int, int]] = None,
        timing_constraints: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[List[str], np.ndarray]]:
        """
        Generate a group of sequences with optional constraints.
        
        This is the main method for generating sequences for simulation groups.
        It supports various constraints needed for different simulation strands.
        
        Parameters
        ----------
        n_sequences : int
            Number of sequences to generate
        dss_template : List[str], optional
            Template DSS pattern. If provided, all sequences use this DSS.
            If None, each sequence gets a random DSS.
        duration_constraints : Dict[int, int], optional
            Constraints on spell durations. Format: {spell_index: fixed_duration}
            spell_index is 0-based (Pythonic indexing).
            Example: {1: 4} means spell at index 1 (second spell) must have duration 4.
        timing_constraints : Dict[str, int], optional
            Constraints on timing. Format: {state_label: time_point}
            Example: {'c': 7} means state 'c' must cover time point 7.
            Not yet fully implemented - placeholder for future extension.
        
        Returns
        -------
        List[Tuple[List[str], np.ndarray]]
            List of (DSS, durations) tuples
        """
        if timing_constraints is not None:
            raise NotImplementedError(
                "timing_constraints is not implemented in generate_group(). "
                "Use sample_sequence_with_focal_state_at_time() instead."
            )
        
        sequences = []
        
        for _ in range(n_sequences):
            # Determine DSS
            if dss_template is not None:
                dss = dss_template.copy()
                if any(a == b for a, b in zip(dss, dss[1:])):
                    raise ValueError(
                        f"dss_template contains adjacent duplicate states: {dss}"
                    )
                # Validate dss_template labels
                invalid_labels = [s for s in dss if s not in self.state_labels]
                if invalid_labels:
                    raise ValueError(
                        f"Invalid state labels in dss_template: {invalid_labels}. "
                        f"Valid labels: {self.state_labels}"
                    )
            else:
                # Estimate number of spells (can be made configurable)
                # If duration_constraints are provided, ensure n_spells is at least
                # (max_idx + 1) to accommodate all constrained indices
                min_n_spells = 3
                if duration_constraints is not None and len(duration_constraints) > 0:
                    max_constraint_idx = max(duration_constraints.keys())
                    min_n_spells = max(min_n_spells, max_constraint_idx + 1)
                
                # Bound complexity: limit max spells based on total_length
                # Each spell needs at least 1 time unit, so max_n_spells <= total_length
                # Also bound by n_states + 1 to avoid excessive complexity
                # Note: This design choice may affect sensitivity conclusions across different n_states
                max_n_spells = min(self.total_length, self.n_states + 1)
                if min_n_spells > max_n_spells:
                    raise ValueError(
                        f"Cannot satisfy duration_constraints: requires at least {min_n_spells} spells, "
                        f"but maximum allowed is {max_n_spells} (design choice to bound complexity)"
                    )
                
                n_spells = self.rng.integers(min_n_spells, max_n_spells + 1)
                dss = self.generate_random_dss(n_spells)
            
            n_spells = len(dss)
            
            # Generate durations with constraints
            if duration_constraints is None:
                durations = self.generate_random_durations(n_spells)
            else:
                # Allocate fixed durations first, then randomize the rest
                durations = np.zeros(n_spells, dtype=int)
                remaining_length = self.total_length
                
                # Set constrained durations
                for idx, fixed_dur in duration_constraints.items():
                    # Explicitly reject negative indices for reproducibility
                    if idx < 0 or idx >= n_spells:
                        raise ValueError(
                            f"Spell index {idx} out of range [0, {n_spells-1}]. "
                            f"Negative indices are not allowed for reproducibility."
                        )
                    if fixed_dur < 1:
                        raise ValueError(f"Duration at index {idx} must be >= 1, got {fixed_dur}")
                    durations[idx] = fixed_dur
                    remaining_length -= fixed_dur
                
                # Check if constrained durations sum exceeds total_length
                if remaining_length < 0:
                    raise ValueError(
                        f"Sum of constrained durations ({self.total_length - remaining_length}) "
                        f"exceeds total_length ({self.total_length})"
                    )
                
                # Check feasibility
                n_unconstrained = n_spells - len(duration_constraints)
                if remaining_length < n_unconstrained:
                    raise ValueError(
                        f"Cannot satisfy duration constraints: "
                        f"remaining_length={remaining_length} < n_unconstrained={n_unconstrained}"
                    )
                
                # Generate random durations for unconstrained spells
                unconstrained_indices = [i for i in range(n_spells) if i not in duration_constraints]
                if len(unconstrained_indices) > 0:
                    # Generate random partition of remaining_length into k positive integers
                    unconstrained_durations = self.generate_random_durations(
                        len(unconstrained_indices), 
                        total_length=remaining_length
                    )
                    
                    for idx, dur in zip(unconstrained_indices, unconstrained_durations):
                        durations[idx] = dur
                else:
                    # All spells are constrained - verify sum
                    if durations.sum() != self.total_length:
                        raise ValueError(
                            f"Constrained durations sum to {durations.sum()}, "
                            f"expected {self.total_length}"
                        )
            
            sequences.append((dss, durations))
        
        return sequences
    
    def _compress_dss(self, dss: List[str], durations: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Merge adjacent identical states in DSS to maintain the invariant.
        
        After relabeling or modifying spells, adjacent identical states may occur.
        This method merges them by summing their durations.
        
        Parameters
        ----------
        dss : List[str]
            DSS that may contain adjacent duplicates
        durations : np.ndarray
            Corresponding durations
        
        Returns
        -------
        Tuple[List[str], np.ndarray]
            Compressed (DSS, durations) with no adjacent duplicates
        
        Raises
        ------
        ValueError
            If dss and durations have inconsistent lengths
        """
        if len(dss) != len(durations):
            raise ValueError(
                f"DSS length {len(dss)} != durations length {len(durations)}"
            )
        
        if len(dss) == 0:
            return (dss.copy(), durations.copy())
        
        compressed_dss = []
        compressed_durations = []
        
        durations = self._validate_integer_durations(durations)
        current_state = dss[0]
        current_duration = int(durations[0])
        
        for i in range(1, len(dss)):
            if dss[i] == current_state:
                # Merge: add duration
                current_duration += int(durations[i])
            else:
                # New state: save previous and start new
                compressed_dss.append(current_state)
                compressed_durations.append(current_duration)
                current_state = dss[i]
                current_duration = int(durations[i])
        
        # Don't forget the last spell
        compressed_dss.append(current_state)
        compressed_durations.append(current_duration)
        
        return (compressed_dss, np.array(compressed_durations, dtype=int))
    
    def sample_sequence_with_focal_state_at_time(
        self,
        dss_template: List[str],
        focal_state: str,
        target_t: int,
        max_attempts: int = 500,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Draw spell durations until the focal state covers ``target_t`` without changing DSS.

        Studer & Ritschard (2016) timing strands keep the DSS template fixed (e.g. ``abcde``
        or ``edcba``) and vary only when the focal state occurs in calendar time. This
        implementation resamples durations only; it never relabels spells.
        """
        if focal_state not in dss_template:
            raise ValueError(
                f"focal_state '{focal_state}' must appear in dss_template {dss_template}"
            )
        if dss_template.count(focal_state) != 1:
            raise ValueError(
                f"Timing DSS template must contain focal_state '{focal_state}' "
                f"exactly once; got {dss_template}."
            )
        if target_t < 0 or target_t >= self.total_length:
            raise ValueError(
                f"target_t {target_t} out of range [0, {self.total_length - 1}]"
            )

        template = list(dss_template)
        n_spells = len(template)

        for _ in range(max_attempts):
            durations = self.generate_random_durations(n_spells)
            expanded = self.expand_to_position_sequence(template, durations)
            if expanded[target_t] == focal_state:
                return template, durations

        raise RuntimeError(
            f"Unable to sample durations such that '{focal_state}' covers t={target_t} "
            f"for template {template} within {max_attempts} attempts."
        )

    def apply_timing_constraint(
        self,
        dss: List[str],
        durations: np.ndarray,
        state_label: str,
        time_point: int,
        on_conflict: str = "raise",
        max_resample_attempts: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """
        Deprecated alias: use :meth:`sample_sequence_with_focal_state_at_time`.

        Resamples durations on a fixed DSS template until ``state_label`` covers
        ``time_point``. Does not relabel spells (Studer-style timing strand).
        
        Parameters
        ----------
        dss : List[str]
            Original DSS
        durations : np.ndarray
            Original durations
        state_label : str
            State that must cover time_point
        time_point : int
            Target time point (0-indexed, must be in [0, total_length-1])
        on_conflict : str, default "raise"
            - "raise": raise if focal-state resampling fails
            - "resample": retry duration resampling up to max_resample_attempts
            - "skip": return the input unchanged when resampling fails
        max_resample_attempts : int
            Maximum duration-resampling attempts when on_conflict="resample".
        
        Returns
        -------
        Tuple[List[str], np.ndarray]
            DSS is unchanged; durations are resampled until the focal state covers
            ``time_point``.
        
        Raises
        ------
        ValueError
            If validation fails, the focal state is absent from the template, or
            resampling fails under on_conflict="raise".
        """
        # Input validation
        if len(dss) != len(durations):
            raise ValueError(
                f"DSS length {len(dss)} != durations length {len(durations)}"
            )
        
        if durations.sum() != self.total_length:
            raise ValueError(
                f"Durations sum to {durations.sum()}, expected {self.total_length}"
            )
        
        if np.any(durations <= 0):
            raise ValueError(
                f"All durations must be >= 1, got: {durations}"
            )
        
        if state_label not in self.state_labels:
            raise ValueError(
                f"state_label '{state_label}' not in valid state labels: {self.state_labels}"
            )
        
        if time_point < 0 or time_point >= self.total_length:
            raise ValueError(f"time_point {time_point} out of range [0, {self.total_length-1}]")
        
        if state_label not in dss:
            if on_conflict == "skip":
                return dss.copy(), durations.copy()
            raise ValueError(
                f"state_label '{state_label}' not in DSS {dss}; timing strands require a "
                "fixed template that already contains the focal state."
            )

        attempts = max_resample_attempts if on_conflict == "resample" else 1
        try:
            return self.sample_sequence_with_focal_state_at_time(
                dss_template=dss,
                focal_state=state_label,
                target_t=time_point,
                max_attempts=attempts,
            )
        except RuntimeError as exc:
            if on_conflict == "skip":
                return dss.copy(), durations.copy()
            raise ValueError(str(exc)) from exc

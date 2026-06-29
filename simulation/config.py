"""
@Author  : Yuqi Liang 梁彧祺
@File    : config.py
@Time    : 05/01/2026 08:40
@Desc    :
Configuration module for sensitivity analysis simulations.

This module defines all configuration parameters, constants, and default values
used throughout the simulation framework. Centralizing configuration here makes
it easy to adjust simulation parameters and ensures consistency across modules.
"""

import math
import warnings

import numpy as np
from typing import List, Tuple, Dict, Any

# expcost (lambda): duration-weight parameter.
# In original OMspell it weights raw-unit expansion terms.
# In OMspellRS, LCPspell, and RLCPspell it weights reference-scaled duration terms.
# Swept in simulations (Studer & Ritschard 2016, §3.4.4; δ >= 0).
EXPCOST_MIN = 0.0
EXPCOST_MAX = None  # no hard upper bound

ALLOWED_DISTANCE_NORMS = ("none", "auto", "ElzingaStuder")
ALLOWED_DIRECTIONAL_DURATION_MODES = (
    "matched",
    "shared_spell_mismatch_compensated",
    "shared_spell_mismatch",  # deprecated alias
    "background_noise",
)

# duration_ref (tau): design-based reference scale — duration differences are expressed
# relative to the total observation window T. Fixed to total_length in simulations;
# not swept (unlike expcost). Matches sequenzo default (number of time positions).


class SimulationConfig:
    """
    Configuration class for sensitivity analysis simulations.
    
    This class holds all parameters needed for the simulation study, including
    sequence generation parameters, distance computation settings, and evaluation
    criteria. All parameters can be overridden when creating an instance.
    """
    
    def __init__(
        self,
        # Sequence generation parameters
        total_length: int = 20,
        n_states: int = 5,
        state_labels: List[str] = None,
        
        # Sample size and replication
        n_sequences_per_group: int = 2000,
        n_replications: int = 30,
        
        # Distance computation parameters
        expcost_values: List[float] = None,  # expcost for LCPspell/RLCPspell/OMspell/OMspellRS
        lcpspell_params: List[float] = None,  # deprecated alias for expcost_values
        om_indel_cost: float = 1.0,
        om_substitution_cost: str = "CONSTANT",
        distance_norm: str = "none",
        directional_duration_mode: str = "matched",
        directional_pair_chunk_size: int = 100,
        log_distance_calls: bool = False,
        
        # Random seed
        random_seed: int = 42,
        
        # Output settings
        output_dir: str = "./results",
        save_sequences: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize simulation configuration.
        
        Parameters
        ----------
        total_length : int
            Fixed total length T for all sequences (default: 20)
        n_states : int
            Number of distinct states in the alphabet (default: 5)
        state_labels : List[str], optional
            Labels for states (e.g., ['a', 'b', 'c', 'd', 'e']). 
            If None, generates labels automatically.
        n_sequences_per_group : int
            Number of sequences to generate per group (default: 2000)
        n_replications : int
            Number of independent replications per simulation strand (default: 30)
        expcost_values : List[float], optional
            expcost values for LCPspell, RLCPspell, OMspell, OMspellRS. Each value must be
            >= 0 (Studer & Ritschard 2016 use e = 0, 0.1, 0.5, 1). Default: [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2].
            duration_ref is fixed to total_length (observation window T) and is not swept.
        lcpspell_params : List[float], optional
            Deprecated alias for ``expcost_values``.
        om_indel_cost : float
            Insertion/deletion cost for OM distance (default: 1.0)
        om_substitution_cost : str
            Substitution cost method for OM (default: "CONSTANT")
        distance_norm : str
            Normalization passed to sequenzo distance computation.
            Use "none" for raw distances (main simulation text),
            "auto" for method-specific built-in normalization (appendix robustness),
            or "ElzingaStuder" for reference-based rescaling (appendix robustness;
            uses a fixed synthetic reference independent of simulated groups).
        directional_duration_mode : str
            Duration structure for spell-order directional strands (4--7 spell panels).
            ``matched`` (default): identical durations within each pair.
            ``shared_spell_mismatch_compensated``: one shared spell gains one time
            unit and another spell loses one unit so total length stays fixed.
            ``background_noise``: independent random durations per trajectory.
            Calendar-time directional strands ignore this setting.
        directional_pair_chunk_size : int
            Number of matched early/late draws per distance-matrix batch in Study 2.
        log_distance_calls : bool
            If True, print one line per distance-matrix computation (debug only).
        random_seed : int
            Base random seed for reproducibility (default: 42)
        output_dir : str
            Directory for saving results (default: "./results")
        save_sequences : bool
            Whether to save generated sequences to disk (default: False)
        verbose : bool
            Whether to print progress messages (default: True)
        """
        if total_length < 1:
            raise ValueError("total_length must be >= 1.")
        if n_states < 1:
            raise ValueError("n_states must be >= 1.")
        if n_sequences_per_group < 1:
            raise ValueError("n_sequences_per_group must be >= 1.")
        if n_replications < 1:
            raise ValueError("n_replications must be >= 1.")

        if expcost_values is not None and lcpspell_params is not None:
            raise ValueError(
                "Specify only expcost_values. lcpspell_params is a deprecated alias "
                "and must not be used together with expcost_values."
            )
        if lcpspell_params is not None:
            warnings.warn(
                "lcpspell_params is deprecated; use expcost_values instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.total_length = total_length
        self.n_states = n_states

        if state_labels is None:
            self.state_labels = [chr(ord('a') + i) for i in range(n_states)]
        else:
            self.state_labels = list(state_labels)
            self.n_states = len(self.state_labels)

        if len(self.state_labels) < 1:
            raise ValueError("state_labels must contain at least one state.")

        if len(set(self.state_labels)) != len(self.state_labels):
            raise ValueError("state_labels must be unique.")

        self.n_sequences_per_group = n_sequences_per_group
        self.n_replications = n_replications

        _expcost = expcost_values if expcost_values is not None else lcpspell_params
        if _expcost is None:
            _expcost = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        if len(_expcost) == 0:
            raise ValueError("expcost_values must not be empty.")

        invalid = [
            p for p in _expcost
            if not isinstance(p, (int, float))
            or not math.isfinite(float(p))
            or float(p) < EXPCOST_MIN
        ]
        if invalid:
            raise ValueError(
                f"expcost_values must contain finite values >= {EXPCOST_MIN}. "
                f"Invalid: {invalid}"
            )
        self.expcost_values = [float(p) for p in _expcost]
        formatted_expcost = [f"{p:.2f}" for p in self.expcost_values]
        for raw, formatted in zip(self.expcost_values, formatted_expcost):
            if not math.isclose(raw, float(formatted), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "Each expcost value must have at most two decimal places because "
                    f"method keys use two-decimal formatting. Got {raw}."
                )
        if len(set(formatted_expcost)) != len(formatted_expcost):
            raise ValueError(
                "expcost_values produce duplicate method keys after two-decimal formatting: "
                f"{formatted_expcost}"
            )
        self.lcpspell_params = self.expcost_values
        
        self.om_indel_cost = om_indel_cost
        self.om_substitution_cost = om_substitution_cost
        if distance_norm not in ALLOWED_DISTANCE_NORMS:
            raise ValueError(
                f"distance_norm must be one of {ALLOWED_DISTANCE_NORMS}, "
                f"got {distance_norm!r}."
            )
        self.distance_norm = distance_norm
        if directional_duration_mode not in ALLOWED_DIRECTIONAL_DURATION_MODES:
            raise ValueError(
                f"directional_duration_mode must be one of "
                f"{ALLOWED_DIRECTIONAL_DURATION_MODES}, got {directional_duration_mode!r}."
            )
        if directional_duration_mode == "shared_spell_mismatch":
            warnings.warn(
                "shared_spell_mismatch is deprecated; use "
                "shared_spell_mismatch_compensated.",
                DeprecationWarning,
                stacklevel=2,
            )
            directional_duration_mode = "shared_spell_mismatch_compensated"
        if directional_pair_chunk_size < 1:
            raise ValueError("directional_pair_chunk_size must be >= 1.")
        self.directional_duration_mode = directional_duration_mode
        self.directional_pair_chunk_size = directional_pair_chunk_size
        self.log_distance_calls = log_distance_calls
        
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.save_sequences = save_sequences
        self.verbose = verbose
    
    def get_distance_methods(self) -> List[str]:
        """
        Get list of all distance methods to evaluate.
        
        This single list is used for every simulation strand (timing, sequencing,
        duration, and the four Study~2 early-vs-late directional strands), so
        LCPspell and RLCPspell appear in
        all strands with every parameter in expcost_values.
        
        Returns
        -------
        List[str]
            List of distance method names, including parameterized variants
            for LCPspell, RLCPspell, OMspell, OMspellRS.
        """
        methods = [
            "HAM",  # Hamming distance (sequenzo uses "HAM" not "Hamming")
            "OM",
            "LCP",
            "RLCP",
            "LCPmst",   # DSS-based LCP with minimal shared time
            "RLCPmst",  # Reversed LCPmst
        ]
        
        # Add LCPspell and RLCPspell with all duration-weight configs (used in every strand)
        # Use consistent formatting (2 decimal places) to ensure matching
        for param in self.expcost_values:
            param_str = f"{param:.2f}"  # Format to 2 decimal places for consistency
            methods.append(f"LCPspell_expcost_{param_str}")
            methods.append(f"RLCPspell_expcost_{param_str}")
        
        # Add OMspell and OMspellRS with all expcost values (including 0)
        for param in self.expcost_values:
            param_str = f"{param:.2f}"  # Format to 2 decimal places for consistency
            methods.append(f"OMspell_expcost_{param_str}")
            methods.append(f"OMspellRS_expcost_{param_str}")
        
        return methods
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for logging/saving.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of configuration
        """
        return {
            "total_length": self.total_length,
            "n_states": self.n_states,
            "state_labels": self.state_labels,
            "n_sequences_per_group": self.n_sequences_per_group,
            "n_replications": self.n_replications,
            "expcost_values": self.expcost_values,
            "lcpspell_params": self.lcpspell_params,
            "om_indel_cost": self.om_indel_cost,
            "om_substitution_cost": self.om_substitution_cost,
            "distance_norm": self.distance_norm,
            "directional_duration_mode": self.directional_duration_mode,
            "directional_pair_chunk_size": self.directional_pair_chunk_size,
            "log_distance_calls": self.log_distance_calls,
            "duration_ref": self.total_length,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
        }

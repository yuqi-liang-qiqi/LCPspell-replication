"""
@Author  : Yuqi Liang 梁彧祺
@File    : distance_computer.py
@Time    : 14/01/2026 15:55
@Desc    :
Distance computation module for sensitivity analysis.

This module provides a unified interface for computing various sequence distances,
using the sequenzo package. It handles conversion between spell-based and 
position-based representations as needed.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

try:
    from sequenzo import SequenceData, get_distance_matrix
except ImportError as e:
    raise ImportError(
        "Could not import sequenzo package from the current Python environment. "
        "Please install it first (for example: pip install sequenzo). "
        f"Original error: {e}"
    )

from .elzinga_reference import build_fixed_elzinga_reference

SpellSeq = Tuple[List[str], np.ndarray]

_NEGATIVE_ALLOWED_METHODS = frozenset({"LCPprod", "RLCPprod"})


def validate_distance_matrix(
    dist_matrix: np.ndarray,
    expected_n: int,
    *,
    method: str,
    allow_negative: bool = False,
) -> np.ndarray:
    """Validate a full pairwise distance matrix returned by Sequenzo."""
    dist = np.asarray(dist_matrix, dtype=np.float64)
    if dist.shape != (expected_n, expected_n):
        raise ValueError(
            f"Distance matrix for method '{method}' has shape {dist.shape}, "
            f"expected ({expected_n}, {expected_n})."
        )
    if not np.all(np.isfinite(dist)):
        raise ValueError(f"Distance matrix for method '{method}' contains non-finite values.")
    if not np.allclose(dist, dist.T, rtol=0.0, atol=1e-8):
        raise ValueError(f"Distance matrix for method '{method}' is not symmetric.")
    if not np.allclose(np.diag(dist), 0.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"Distance matrix for method '{method}' has non-zero diagonal.")
    if not allow_negative and np.any(dist < -1e-8):
        raise ValueError(f"Distance matrix for method '{method}' contains negative values.")
    return dist


def validate_spell_sequence(
    dss: List[str],
    durations: np.ndarray,
    *,
    total_length: int,
    state_labels: List[str],
    sequence_index: Optional[int] = None,
) -> np.ndarray:
    """Reject invalid spell sequences before distance computation."""
    prefix = f"Sequence {sequence_index}: " if sequence_index is not None else ""
    if len(dss) != len(durations):
        raise ValueError(
            f"{prefix}DSS length {len(dss)} != durations length {len(durations)}."
        )
    raw_dur = np.asarray(durations)
    if not np.issubdtype(raw_dur.dtype, np.number):
        raise ValueError(f"{prefix}Spell durations must be numeric.")
    if not np.all(np.isfinite(raw_dur)):
        raise ValueError(f"{prefix}Spell durations must be finite.")
    if not np.all(raw_dur == np.floor(raw_dur)):
        raise ValueError(f"{prefix}Spell durations must be integers.")
    dur = raw_dur.astype(int)
    if np.any(dur <= 0):
        raise ValueError(f"{prefix}All spell durations must be positive.")
    if int(dur.sum()) != total_length:
        raise ValueError(
            f"{prefix}Durations sum to {int(dur.sum())}, expected {total_length}."
        )
    if any(a == b for a, b in zip(dss, dss[1:])):
        raise ValueError(
            f"{prefix}DSS contains adjacent duplicate states: {dss}."
        )
    for state in dss:
        if state not in state_labels:
            raise ValueError(
                f"{prefix}State '{state}' not in alphabet {state_labels}."
            )
    return dur


class DistanceComputer:
    """
    Unified interface for computing sequence distances.
    
    This class handles the conversion between spell-based representations
    (used for generation) and position-based representations (used by most
    distance functions), and provides a consistent API for all distance methods.
    """
    
    def __init__(self, state_labels: List[str], total_length: int, default_norm: str = "none"):
        """
        Initialize distance computer.
        
        Parameters
        ----------
        state_labels : List[str]
            List of state labels (alphabet)
        total_length : int
            Fixed total length T for all sequences
        default_norm : str
            Default normalization mode forwarded to sequenzo when no explicit
            norm is passed in compute_distance_matrix.
        """
        self.state_labels = state_labels
        self.state_to_int = {label: i for i, label in enumerate(state_labels)}
        self.int_to_state = {i: label for i, label in enumerate(state_labels)}
        self.total_length = total_length
        self.default_norm = default_norm
        
        # Define allowed methods (whitelist for safety)
        self.allowed_methods = {
            "HAM", "OM", "LCP", "RLCP", 
            "LCPmst", "RLCPmst", "LCPprod", "RLCPprod",
            "LCPspell", "RLCPspell", "OMspell", "OMspellRS",
            "DHD", "CHI2", "EUCLID", "NMS", "NMSMST", "SVRspell"
        }

    def _fixed_elzinga_reference(self) -> SpellSeq:
        return build_fixed_elzinga_reference(self.state_labels, self.total_length)
    
    def sequences_to_seqdata(
        self,
        sequences: List[Tuple[List[str], np.ndarray]]
    ) -> SequenceData:
        """
        Convert list of (DSS, durations) tuples to SequenceData object.
        """
        position_sequences = []
        for idx, (dss, durations) in enumerate(sequences):
            durations = validate_spell_sequence(
                dss,
                durations,
                total_length=self.total_length,
                state_labels=self.state_labels,
                sequence_index=idx,
            )

            seq = []
            for state, duration in zip(dss, durations):
                seq.extend([self.state_to_int[state]] * int(duration))
            
            if len(seq) != self.total_length:
                raise ValueError(
                    f"Sequence {idx}: expanded length {len(seq)} != total_length {self.total_length}. "
                    f"Durations sum to {durations.sum()}. This indicates a bug in expansion logic. "
                    f"Do not pad or truncate - fix the generation/expansion code instead."
                )
            
            position_sequences.append(seq)
        
        import pandas as pd
        
        position_sequences_labels = []
        for seq in position_sequences:
            seq_labels = [self.int_to_state[code] for code in seq]
            position_sequences_labels.append(seq_labels)
        
        time_cols = [f'T{i+1}' for i in range(self.total_length)]
        df = pd.DataFrame(position_sequences_labels, columns=time_cols)
        
        seqdata = SequenceData(
            data=df,
            time=time_cols,
            states=self.state_labels
        )
        
        return seqdata
    
    def compute_distance_matrix(
        self,
        sequences: List[Tuple[List[str], np.ndarray]],
        method: str,
        expcost: Optional[float] = None,
        norm: Optional[str] = None,
        om_indel_cost: Optional[float] = None,
        om_substitution_cost: Optional[str] = None,
        strand_name: Optional[str] = None,
        replication_id: Optional[int] = None,
        log_distance: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix for a set of sequences.
        """
        if log_distance:
            params_str = []
            if expcost is not None:
                params_str.append(f"expcost={expcost}")
            if om_indel_cost is not None:
                params_str.append(f"indel={om_indel_cost}")
            if om_substitution_cost is not None:
                params_str.append(f"subst={om_substitution_cost}")
            params_display = ", ".join(params_str) if params_str else "default"
            strand_display = strand_name if strand_name is not None else "unknown"
            rep_display = replication_id if replication_id is not None else "?"
            print(
                f"[DISTANCE_COMPUTE] strand={strand_display}, rep={rep_display}, "
                f"method={method}, n={len(sequences)}, params=({params_display})"
            )
        
        if norm is None:
            norm = self.default_norm

        if method not in self.allowed_methods:
            raise ValueError(
                f"Method '{method}' is not in the allowed list. "
                f"Allowed methods: {sorted(self.allowed_methods)}"
            )
        
        n_eval = len(sequences)
        sequences_input = list(sequences)
        elzinga_ref_index: Optional[int] = None
        if norm == "ElzingaStuder":
            ref_seq = self._fixed_elzinga_reference()
            elzinga_ref_index = n_eval
            sequences_input = sequences_input + [ref_seq]
            kwargs = dict(kwargs)
            kwargs["normalization_reference_index"] = elzinga_ref_index
        
        seqdata = self.sequences_to_seqdata(sequences_input)
        
        if method in {"LCPspell", "RLCPspell", "OMspell", "OMspellRS"} and expcost is None:
            raise ValueError(
                f"expcost must be provided explicitly for method '{method}' "
                "in the replication pipeline."
            )
        
        method_kwargs = {}
        duration_ref = float(self.total_length)
        
        if method == "OM":
            method_kwargs["sm"] = om_substitution_cost if om_substitution_cost is not None else "CONSTANT"
            method_kwargs["indel"] = om_indel_cost if om_indel_cost is not None else 1.0
        
        if method in ["LCPspell", "RLCPspell", "OMspellRS"]:
            method_kwargs["duration_ref"] = duration_ref
        
        if method in ["LCPspell", "RLCPspell"]:
            method_kwargs["expcost"] = expcost
        
        if method in ["OMspell", "OMspellRS"]:
            method_kwargs["expcost"] = expcost
            method_kwargs["sm"] = om_substitution_cost if om_substitution_cost is not None else "CONSTANT"
            method_kwargs["indel"] = om_indel_cost if om_indel_cost is not None else 1.0
        
        method_kwargs.update(kwargs)
        
        try:
            dist_matrix = get_distance_matrix(
                seqdata=seqdata,
                method=method,
                norm=norm,
                full_matrix=True,
                **method_kwargs
            )
        except Exception as e:
            raise ValueError(
                f"Failed to compute distance matrix for method '{method}'. "
                f"This method may not be available in sequenzo or parameters are invalid. "
                f"Original error: {str(e)}"
            ) from e
        
        if hasattr(dist_matrix, 'values'):
            dist_matrix = dist_matrix.values
        
        dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
        
        if elzinga_ref_index is not None:
            dist_matrix = dist_matrix[:n_eval, :n_eval]

        allow_negative = method in _NEGATIVE_ALLOWED_METHODS
        return validate_distance_matrix(
            dist_matrix,
            n_eval,
            method=method,
            allow_negative=allow_negative,
        )
    
    def compute_all_distances(
        self,
        sequences: List[Tuple[List[str], np.ndarray]],
        methods: List[str],
        om_indel_cost: Optional[float] = None,
        om_substitution_cost: Optional[str] = None,
        strand_name: Optional[str] = None,
        replication_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute distance matrices for multiple methods.
        """
        results = {}
        failed_methods = []
        
        for method in methods:
            try:
                if (method.startswith("LCPspell_expcost_") or method.startswith("RLCPspell_expcost_") or
                    method.startswith("OMspell_expcost_") or method.startswith("OMspellRS_expcost_")):
                    expcost = float(method.split("_")[-1])
                    base_method = method.split("_expcost_")[0]
                    dist_matrix = self.compute_distance_matrix(
                        sequences, base_method, expcost=expcost,
                        om_indel_cost=om_indel_cost,
                        om_substitution_cost=om_substitution_cost,
                        strand_name=strand_name,
                        replication_id=replication_id,
                        **kwargs
                    )
                else:
                    dist_matrix = self.compute_distance_matrix(
                        sequences, method,
                        om_indel_cost=om_indel_cost,
                        om_substitution_cost=om_substitution_cost,
                        strand_name=strand_name,
                        replication_id=replication_id,
                        **kwargs
                    )
                
                results[method] = dist_matrix
            except Exception as e:
                failed_methods.append((method, str(e)))
                continue
        
        if failed_methods:
            error_msg = "Failed to compute distances for some methods:\n"
            for method, error in failed_methods:
                error_msg += f"  - {method}: {error}\n"
            raise ValueError(error_msg)
        
        return results
    
    def validate_methods(self, methods: List[str]) -> Tuple[List[str], List[str]]:
        valid = []
        invalid = []
        
        for method in methods:
            if (method.startswith("LCPspell_expcost_") or method.startswith("RLCPspell_expcost_") or
                method.startswith("OMspell_expcost_") or method.startswith("OMspellRS_expcost_")):
                base_method = method.split("_expcost_")[0]
            else:
                base_method = method
            
            if base_method in self.allowed_methods:
                valid.append(method)
            else:
                invalid.append(method)
        
        return (valid, invalid)

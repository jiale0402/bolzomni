"""Score unrepaired joint outputs separately from MPNN scaffold redesignability.

No model is imported here. Invalid chemistry stays in the denominator and is
never replaced by a different sequence before raw scoring (draft section 4.4).
"""
import math

CANONICAL = frozenset('ACDEFGHIKLMNPQRSTVWY')


def chemical_validity(sequence, geometric_sequence, expected_observed, *, finite_geometry):
    """expected_observed maps one-based residue positions to observed letters."""
    length_ok = len(sequence) == len(geometric_sequence) and bool(sequence)
    identity_bad = [i for i, aa in enumerate(sequence, 1) if aa not in CANONICAL]
    geometry_bad = [i for i, aa in enumerate(geometric_sequence, 1) if aa not in CANONICAL]
    mismatch = [i for i, (a, b) in enumerate(zip(sequence, geometric_sequence), 1) if a != b]
    observed_bad = [int(i) for i, aa in expected_observed.items()
        if int(i) < 1 or int(i) > len(sequence) or sequence[int(i)-1] != aa]
    valid = bool(finite_geometry and length_ok and not any(
        (identity_bad, geometry_bad, mismatch, observed_bad)))
    return dict(valid=valid, finite_geometry=bool(finite_geometry), length_consistent=length_ok,
        invalid_identity_positions=identity_bad, invalid_geometry_positions=geometry_bad,
        identity_geometry_mismatches=mismatch, observed_identity_violations=observed_bad,
        scope='Chemistry/geometry consistency only; physical quality requires independent refolding')


def design_pass(metrics, thresholds):
    """Same four criteria as the locked benchmark, applied to one raw sequence."""
    keys=('rmsd','motif_rmsd','plddt','pae')
    if any(metrics.get(k) is None or not math.isfinite(float(metrics[k])) for k in keys):
        return False
    return (metrics['rmsd'] <= thresholds['ca_rmsd_max']
        and metrics['motif_rmsd'] <= thresholds['motif_rmsd_max']
        and metrics['plddt'] >= thresholds['plddt_min']
        and metrics['pae'] <= thresholds['pae_max'])

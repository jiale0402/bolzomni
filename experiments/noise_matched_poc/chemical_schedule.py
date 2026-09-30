"""Paper Algorithm 1 chemistry, scheduled by coordinate-training noise coverage.

Pure scalar schedule construction. No model import, checkpoint access or compute
submission. This is a candidate sampling choice, not a verified performance fix.
"""
import math


def noise_cdf(sigma, sigma_data, p_mean, p_std):
    if not all(math.isfinite(v) for v in (sigma, sigma_data, p_mean, p_std)):
        raise ValueError('Schedule parameters must be finite')
    if sigma < 0 or sigma_data <= 0 or p_std <= 0:
        raise ValueError('Invalid noise distribution parameters')
    if sigma == 0:
        return 0.0
    z = (math.log(sigma / sigma_data) - p_mean) / p_std
    return 0.5 * math.erfc(-z / math.sqrt(2))


def chemical_levels(denoiser_sigmas, *, sigma_data, p_mean, p_std):
    """Return K+1 mask levels, with t[0]=1 and t[K]=0.

The input is the K actual noise levels passed to the denoiser, including native
churn, not just the base solver schedule. Repeats are legal and reveal nothing
between equal levels. Unsupported increasing levels fail rather than silently
changing the native coordinate solver.
"""
    sigmas = [float(s) for s in denoiser_sigmas]
    if not sigmas or any(not math.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError('Require positive finite denoiser noise levels')
    if any(b > a for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError('Denoiser sigma schedule must be nonincreasing')
    probabilities = [noise_cdf(s, sigma_data, p_mean, p_std) for s in sigmas]
    if probabilities[0] <= 0 or probabilities[-1] <= 0:
        raise ValueError('CDF underflow prevents a valid absorbing-mask schedule')
    # Normalize the finite initial noise to exactly all MASK. Intermediate values
    # retain the distribution CDF; the native terminal zero is explicitly t=0.
    levels = [p / probabilities[0] for p in probabilities] + [0.0]
    levels[0] = 1.0
    return levels


def reveal_probabilities(levels):
    values = [float(t) for t in levels]
    if len(values) < 2 or values[0] != 1.0 or values[-1] != 0.0:
        raise ValueError('Require exact fully-masked and terminal-unmasked endpoints')
    if any(not math.isfinite(t) or not 0 < t <= 1 for t in values[:-1]):
        raise ValueError('Interior mask levels must be positive and finite')
    if any(b > a for a, b in zip(values, values[1:])):
        raise ValueError('Mask levels must be nonincreasing')
    return [1.0 - b/a for a,b in zip(values,values[1:])]


def joint_objective(parts, *, identity_weight, replay_weight):
    """One actual weighted formula for both training and report reconstruction."""
    if not math.isfinite(identity_weight) or identity_weight <= 0:
        raise ValueError('identity_weight must be finite and positive')
    if not math.isfinite(replay_weight) or replay_weight < 0:
        raise ValueError('replay_weight must be finite and nonnegative')
    return (parts['coordinate'] + parts['smooth_lddt']
            + identity_weight * parts['identity'] + replay_weight * parts['replay'])

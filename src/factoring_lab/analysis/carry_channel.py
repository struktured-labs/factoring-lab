"""Information-theoretic analysis of the carry-propagation channel.

The carry chain t_0, t_1, ..., t_{d-1} in the digit convolution formulation
forms a Markov chain whose transitions are governed by the transfer matrices
T_k.  This module computes:

1. The forward-backward marginals: P(t_k = s) over the uniform distribution
   on lattice points Lambda_n ∩ B.

2. The carry chain entropy: H(T_0, ..., T_{d-1}), which equals the mutual
   information I(Z; T) since T is a deterministic function of the lattice
   point Z.

3. The residual uncertainty: H(Z|T) = log2(R) - H(T), measuring how much
   information is left after conditioning on the carry sequence.

4. The information gap: H(Z|T) = Theta(d^2) while H(T) = O(d log d),
   proving that the carries reveal only a vanishing fraction of the
   information needed to identify a rank-1 point.

Key theorem this supports:
    Even if the carry sequence were fully known, the search space remains
    2^{Theta(d^2)}.  The rank-1 constraint (not the carry chain) is the
    true locus of hardness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    _count_bounded_compositions,
    to_digits,
)


@dataclass
class CarryChannelResult:
    """Information-theoretic analysis of the carry-propagation channel."""

    n: int
    base: int
    d: int
    dx: int
    dy: int

    # Total lattice point count (log2)
    log2_total_lattice_points: float

    # Carry chain entropy H(T) in bits — equals I(Z; T)
    carry_entropy: float

    # Residual uncertainty H(Z|T) = log2(R) - H(T) in bits
    residual_uncertainty: float

    # Fraction of information revealed: H(T) / log2(R)
    fraction_revealed: float

    # Per-position conditional entropy H(t_k | t_{k-1}) in bits
    conditional_entropies: list[float]

    # Marginal carry distributions P(t_k = s) at each position
    marginal_distributions: list[dict[int, float]] = field(repr=False)

    # Per-position fiber sizes: E[log2(|F_t|)] conditioned on t_k
    # (average log-fiber-size at each position)
    avg_log_fiber_sizes: list[float] = field(default_factory=list)

    # For the TRUE factorization: carry sequence and its fiber size
    true_carry_sequence: list[int] | None = None
    true_fiber_log2: float | None = None


def analyze_carry_channel(
    n: int,
    base: int,
    dx: int | None = None,
    dy: int | None = None,
    p: int | None = None,
    q: int | None = None,
) -> CarryChannelResult:
    """Compute information-theoretic properties of the carry channel.

    Uses the forward-backward algorithm on the transfer matrix chain:
      - Forward pass:  alpha[k][s] = #configs for positions 0..k with t_k = s
      - Backward pass: beta[k][s]  = #configs for positions k+1..d-1 with t_k = s, t_{d-1}→0
      - Marginal: P(t_k = s) = alpha[k][s] * beta[k][s] / R
      - Joint: P(t_k = s, t_{k+1} = s') = alpha[k][s] * T_{k+1}[s,s'] * beta[k+1][s'] / R

    The carry chain entropy is computed via the chain rule:
      H(T) = H(t_0) + sum_{k=1}^{d-1} H(t_k | t_{k-1})

    Parameters
    ----------
    n : int
        The semiprime to analyze.
    base : int
        The base for digit representation.
    dx, dy : int, optional
        Factor digit sizes (auto-computed if None).
    p, q : int, optional
        Known factors, if available. Used to compute the true carry sequence
        and its fiber size.
    """
    c = to_digits(n, base)
    d = len(c)

    if dx is None or dy is None:
        _, dx, dy = _compute_digit_sizes(n, base)

    max_z = (base - 1) ** 2

    # Number of z-terms at each position
    num_terms_at: list[int] = []
    for k in range(d):
        count = 0
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy:
                count += 1
        num_terms_at.append(count)

    # Max carry at each position
    max_carry_at: list[int] = []
    max_t = 0
    for k in range(d):
        max_sum = num_terms_at[k] * max_z + max_t
        max_t = max_sum // base
        max_carry_at.append(max_t)

    # Precompute composition counts per position
    # comp_caches[k][target] = count of bounded compositions
    comp_caches: list[dict[int, int]] = []
    for k in range(d):
        max_carry_in = 0 if k == 0 else max_carry_at[k - 1]
        max_carry_out = max_carry_at[k]
        max_target = c[k] + base * max_carry_out
        min_target = max(0, c[k] - max_carry_in)
        cache: dict[int, int] = {}
        for target in range(min_target, max_target + 1):
            cache[target] = _count_bounded_compositions(
                target, num_terms_at[k], max_z
            )
        comp_caches.append(cache)

    # ----------------------------------------------------------------
    # Forward pass: alpha[k][s] = #lattice configs for positions 0..k
    # ending with carry t_k = s.
    # ----------------------------------------------------------------
    forward: list[dict[int, int]] = []

    # Position 0: carry_in = 0
    alpha: dict[int, int] = {}
    for t_out in range(max_carry_at[0] + 1):
        target = c[0] + base * t_out
        cnt = comp_caches[0].get(target, 0)
        if cnt > 0:
            alpha[t_out] = cnt
    forward.append(alpha)

    for k in range(1, d):
        max_carry_out = max_carry_at[k]
        new_alpha: dict[int, int] = {}
        prev = forward[k - 1]
        for t_out in range(max_carry_out + 1):
            total = 0
            for t_in, weight in prev.items():
                target = c[k] - t_in + base * t_out
                if target < 0:
                    continue
                cnt = comp_caches[k].get(target, 0)
                if cnt > 0:
                    total += weight * cnt
            if total > 0:
                new_alpha[t_out] = total
        forward.append(new_alpha)

    # Total count R = forward[d-1][0] (final carry must be 0)
    R = forward[d - 1].get(0, 0)
    log2_R = log2(R) if R > 0 else 0.0

    # ----------------------------------------------------------------
    # Backward pass: beta[k][s] = #lattice configs for positions k+1..d-1
    # starting with carry t_k = s and ending with final carry = 0.
    # ----------------------------------------------------------------
    backward: list[dict[int, int]] = [{}] * d

    # Position d-1: final carry must be 0
    backward[d - 1] = {0: 1}

    for k in range(d - 2, -1, -1):
        # beta[k][s] = sum over s' of T_{k+1}[s, s'] * beta[k+1][s']
        beta_k: dict[int, int] = {}
        beta_next = backward[k + 1]
        max_carry_out_next = max_carry_at[k + 1]
        for t_in in range(max_carry_at[k] + 1):
            total = 0
            for t_out, beta_val in beta_next.items():
                target = c[k + 1] - t_in + base * t_out
                if target < 0:
                    continue
                cnt = comp_caches[k + 1].get(target, 0)
                if cnt > 0:
                    total += cnt * beta_val
            if total > 0:
                beta_k[t_in] = total
        backward[k] = beta_k

    # ----------------------------------------------------------------
    # Marginal distributions P(t_k = s) = alpha[k][s] * beta[k][s] / R
    # ----------------------------------------------------------------
    marginals: list[dict[int, float]] = []
    for k in range(d):
        dist: dict[int, float] = {}
        for s in forward[k]:
            if s in backward[k]:
                prob = (forward[k][s] * backward[k][s]) / R
                if prob > 0:
                    dist[s] = prob
        marginals.append(dist)

    # ----------------------------------------------------------------
    # Carry chain entropy via chain rule:
    #   H(T) = H(t_0) + sum_{k=1}^{d-1} H(t_k | t_{k-1})
    #
    # H(t_k | t_{k-1}) = sum_{s} P(t_{k-1}=s) * H(t_k | t_{k-1}=s)
    # where H(t_k | t_{k-1}=s) = -sum_{s'} P(t_k=s'|t_{k-1}=s) log P(...)
    # ----------------------------------------------------------------
    conditional_entropies: list[float] = []

    # H(t_0): entropy of the marginal at position 0
    h0 = _entropy(marginals[0])
    conditional_entropies.append(h0)

    for k in range(1, d):
        # Compute joint P(t_{k-1}=s, t_k=s') and conditional H(t_k|t_{k-1})
        # P(t_{k-1}=s, t_k=s') = alpha[k-1][s] * T_k[s,s'] * beta[k][s'] / R
        # T_k[s, s'] = comp_caches[k][c[k] - s + base*s'] if target >= 0

        # Group by t_{k-1} to compute conditional entropy
        h_cond = 0.0
        for s, alpha_s in forward[k - 1].items():
            # P(t_{k-1} = s)
            beta_s_prev = backward[k - 1].get(s, 0)
            p_s = (alpha_s * beta_s_prev) / R if R > 0 else 0.0
            if p_s <= 0:
                continue

            # Conditional distribution P(t_k = s' | t_{k-1} = s)
            cond_dist: dict[int, float] = {}
            total_given_s = 0.0
            for s_prime in backward[k]:
                target = c[k] - s + base * s_prime
                if target < 0:
                    continue
                cnt = comp_caches[k].get(target, 0)
                if cnt > 0:
                    beta_sp = backward[k].get(s_prime, 0)
                    joint = cnt * beta_sp
                    if joint > 0:
                        cond_dist[s_prime] = float(joint)
                        total_given_s += joint

            # Normalize and compute entropy
            if total_given_s > 0:
                for s_prime in cond_dist:
                    cond_dist[s_prime] /= total_given_s
                h_cond += p_s * _entropy(cond_dist)

        conditional_entropies.append(h_cond)

    carry_entropy = sum(conditional_entropies)
    residual = log2_R - carry_entropy
    fraction = carry_entropy / log2_R if log2_R > 0 else 0.0

    # ----------------------------------------------------------------
    # Average fiber sizes at each position
    # The fiber at position k given t_k = s has log-size:
    #   log2(alpha[k][s]) + log2(beta[k][s]) - log2(R)  ... no, that's P(t_k=s)
    # Actually, fiber size given FULL carry sequence t = (t_0,...,t_{d-1}):
    #   |F_t| = product_k comp_count(k, t_{k-1}, t_k)
    # The average log-fiber-size is:
    #   E[log2(|F_t|)] = sum_k E[log2(comp_count(k, t_{k-1}, t_k))]
    # ----------------------------------------------------------------
    avg_log_fiber: list[float] = []
    for k in range(d):
        # E[log2(comp_count(k, t_{k-1}, t_k))] over joint (t_{k-1}, t_k)
        avg = 0.0
        if k == 0:
            # t_{-1} = 0, so only sum over t_0
            for s, prob in marginals[0].items():
                target = c[0] + base * s
                cnt = comp_caches[0].get(target, 0)
                if cnt > 0 and prob > 0:
                    avg += prob * log2(cnt)
        else:
            # Sum over joint (t_{k-1}, t_k)
            for s, alpha_s in forward[k - 1].items():
                beta_s_prev = backward[k - 1].get(s, 0)
                p_s = (alpha_s * beta_s_prev) / R if R > 0 else 0.0
                if p_s <= 0:
                    continue
                # Conditional P(t_k=s' | t_{k-1}=s)
                total_given_s = 0
                entries: list[tuple[int, int]] = []
                for s_prime in backward[k]:
                    target = c[k] - s + base * s_prime
                    if target < 0:
                        continue
                    cnt = comp_caches[k].get(target, 0)
                    if cnt > 0:
                        beta_sp = backward[k].get(s_prime, 0)
                        joint = cnt * beta_sp
                        if joint > 0:
                            entries.append((cnt, joint))
                            total_given_s += joint
                if total_given_s > 0:
                    for cnt, joint in entries:
                        p_cond = joint / total_given_s
                        avg += p_s * p_cond * log2(cnt)
        avg_log_fiber.append(avg)

    # ----------------------------------------------------------------
    # True factorization analysis (if p, q provided)
    # ----------------------------------------------------------------
    true_carry_seq: list[int] | None = None
    true_fiber_log2: float | None = None

    if p is not None and q is not None:
        x_digits = to_digits(p, base)
        y_digits = to_digits(q, base)
        # Pad to dx, dy
        while len(x_digits) < dx:
            x_digits.append(0)
        while len(y_digits) < dy:
            y_digits.append(0)

        # Compute carry sequence
        true_carry_seq = []
        carry = 0
        fiber_log2 = 0.0
        for k in range(d):
            conv_sum = 0
            for i in range(min(k + 1, dx)):
                j = k - i
                if 0 <= j < dy:
                    conv_sum += x_digits[i] * y_digits[j]
            total_at_k = conv_sum + carry
            t_k = (total_at_k - c[k]) // base
            true_carry_seq.append(t_k)

            # Fiber count at this position given carry_in and carry_out
            target = c[k] - carry + base * t_k
            cnt = comp_caches[k].get(target, 0)
            if cnt > 0:
                fiber_log2 += log2(cnt)
            carry = t_k

        true_fiber_log2 = fiber_log2

    return CarryChannelResult(
        n=n,
        base=base,
        d=d,
        dx=dx,
        dy=dy,
        log2_total_lattice_points=log2_R,
        carry_entropy=carry_entropy,
        residual_uncertainty=residual,
        fraction_revealed=fraction,
        conditional_entropies=conditional_entropies,
        marginal_distributions=marginals,
        avg_log_fiber_sizes=avg_log_fiber,
        true_carry_sequence=true_carry_seq,
        true_fiber_log2=true_fiber_log2,
    )


def _entropy(dist: dict[int, float]) -> float:
    """Compute entropy of a discrete distribution (in bits)."""
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * log2(p)
    return h

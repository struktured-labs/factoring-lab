//! Rust implementation of digit convolution backtracking factoring.
//!
//! Port of the Python `DigitConvolution` algorithm. Solves digit-level
//! convolution constraints with carry propagation via backtracking.

use pyo3::prelude::*;

/// Convert n to base-b digits, least significant first.
fn to_digits(mut n: u128, base: u32) -> Vec<u32> {
    if n == 0 {
        return vec![0];
    }
    let b = base as u128;
    let mut digits = Vec::new();
    while n > 0 {
        digits.push((n % b) as u32);
        n /= b;
    }
    digits
}

/// Convert base-b digits (LSB first) back to integer.
fn from_digits(digits: &[u32], base: u32) -> u128 {
    let b = base as u128;
    let mut result: u128 = 0;
    for &d in digits.iter().rev() {
        result = result * b + d as u128;
    }
    result
}

/// Internal recursive solver. Returns Some((factor, iterations)) on success.
fn solve(
    k: usize,
    carry: u128,
    x_digits: &mut Vec<u32>,
    y_digits: &mut Vec<u32>,
    c: &[u32],
    d: usize,
    base: u32,
    n: u128,
    iterations: &mut u64,
) -> Option<u128> {
    let b = base as u128;
    *iterations += 1;

    if k == d {
        if carry == 0 {
            let x = from_digits(x_digits, base);
            let y = from_digits(y_digits, base);
            if x > 1 && y > 1 && x * y == n {
                return Some(x.min(y));
            }
        }
        return None;
    }

    // Compute partial sum from previously chosen digits: sum_{i=1}^{k-1} x_i * y_{k-i}
    let mut partial: u128 = 0;
    for i in 1..k {
        partial += x_digits[i] as u128 * y_digits[k - i] as u128;
    }

    let x0 = x_digits[0] as u128;
    let y0 = y_digits[0] as u128;
    let ck = c[k] as u128;

    for xk in 0..base {
        for yk in 0..base {
            let total = partial + x0 * yk as u128 + xk as u128 * y0 + carry;
            if total % b == ck {
                x_digits[k] = xk;
                y_digits[k] = yk;
                let new_carry = total / b;
                if let Some(result) = solve(k + 1, new_carry, x_digits, y_digits, c, d, base, n, iterations) {
                    return Some(result);
                }
            }
        }
    }

    None
}

/// Factor n using digit convolution backtracking in the given base.
///
/// Returns `(factor, cofactor, iterations)` on success, or `None` if no
/// non-trivial factorization is found.
fn digit_convolution_factor_inner(
    n: u128,
    base: u32,
    max_digits: Option<usize>,
) -> (Option<(u128, u128)>, u64) {
    let c = to_digits(n, base);
    let mut d = c.len();
    if let Some(md) = max_digits {
        d = d.min(md);
    }

    let b = base as u128;
    let mut x_digits = vec![0u32; d];
    let mut y_digits = vec![0u32; d];
    let mut iterations: u64 = 0;

    for x0 in 1..base {
        for y0 in x0..base {
            let total = x0 as u128 * y0 as u128;
            if total % b == c[0] as u128 {
                x_digits[0] = x0;
                y_digits[0] = y0;
                let carry = total / b;
                if let Some(factor) = solve(
                    1,
                    carry,
                    &mut x_digits,
                    &mut y_digits,
                    &c,
                    d,
                    base,
                    n,
                    &mut iterations,
                ) {
                    let cofactor = n / factor;
                    return (Some((factor, cofactor)), iterations);
                }
            }
        }
    }

    (None, iterations)
}

/// Factor n using digit convolution backtracking.
///
/// Args:
///     n: The number to factor (up to ~38 decimal digits with u128).
///     base: The base for digit decomposition (default 10).
///     max_digits: Optional cap on the number of digit positions to search.
///
/// Returns:
///     A tuple (factor, cofactor, iterations) if a non-trivial factorization
///     is found, or None otherwise.
#[pyfunction]
#[pyo3(signature = (n, base=10, max_digits=None))]
fn digit_convolution_factor(
    n: u128,
    base: u32,
    max_digits: Option<usize>,
) -> Option<(u128, u128, u64)> {
    // Handle trivial cases
    if n < 4 {
        return None;
    }
    if n % 2 == 0 {
        return Some((2, n / 2, 0));
    }

    let (result, iterations) = digit_convolution_factor_inner(n, base, max_digits);
    result.map(|(f, c)| (f, c, iterations))
}

/// A Python module implemented in Rust.
#[pymodule]
fn factoring_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(digit_convolution_factor, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_semiprimes() {
        let cases = vec![
            (15u128, vec![3, 5]),
            (21, vec![3, 7]),
            (35, vec![5, 7]),
            (77, vec![7, 11]),
            (143, vec![11, 13]),
            (221, vec![13, 17]),
        ];
        for (n, factors) in cases {
            let result = digit_convolution_factor(n, 10, None);
            assert!(result.is_some(), "Failed to factor {}", n);
            let (f, c, _) = result.unwrap();
            assert!(
                factors.contains(&f) || factors.contains(&c),
                "Wrong factors for {}: got ({}, {})",
                n, f, c,
            );
        }
    }

    #[test]
    fn test_base2() {
        let result = digit_convolution_factor(15, 2, None);
        assert!(result.is_some());
        let (f, _, _) = result.unwrap();
        assert!(f == 3 || f == 5);
    }

    #[test]
    fn test_prime_returns_none() {
        let result = digit_convolution_factor(97, 10, None);
        assert!(result.is_none());
    }
}

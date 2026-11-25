use serde::{Deserialize, Serialize};

/// SplitMix64: seedable and serializable, so a resumed checkpoint continues the same stream.
// ponytail: not Python's MT19937, runs don't replay gepa-python step for step; port MT19937 if that's ever required.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [0, n). Panics on n == 0.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below(0)");
        // ponytail: modulo bias is ~n/2^64, irrelevant at candidate-pool sizes
        (self.next_u64() % n as u64) as usize
    }

    pub fn choice<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    /// `k` elements at distinct positions, in random order (Python `random.sample`).
    pub fn sample<T: Copy>(&mut self, xs: &[T], k: usize) -> Vec<T> {
        let mut v = xs.to_vec();
        let k = k.min(v.len());
        for i in 0..k {
            let j = i + self.below(v.len() - i);
            v.swap(i, j);
        }
        v.truncate(k);
        v
    }

    /// Index drawn proportionally to the positive `weights`; uniform when none is positive
    /// (Python `random.choices` raises there instead).
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().filter(|w| **w > 0.0).sum();
        if !(total > 0.0) {
            return self.below(weights.len());
        }
        let mut r = self.f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.0 {
                if r < w {
                    return i;
                }
                r -= w;
            }
        }
        weights.iter().rposition(|w| *w > 0.0).unwrap()
    }
}

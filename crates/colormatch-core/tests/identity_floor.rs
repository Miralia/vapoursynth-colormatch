//! Measures the discretisation floor of the builder: how far a model built from an *identical*
//! reference moves pixels. This bounds the accuracy the sequential sweep can reach at all.
use colormatch_core::apply::apply_point;
use colormatch_core::build::{BuildOptions, Samples, build_model};

fn cloud(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 32) as f32 / u32::MAX as f32
    };
    (0..n)
        .map(|_| {
            [
                (0.3 + 0.28 * (next() * 2.0 - 1.0)).clamp(0.0, 1.0),
                (0.3 + 0.28 * (next() * 2.0 - 1.0)).clamp(0.0, 1.0),
                (0.3 + 0.28 * (next() * 2.0 - 1.0)).clamp(0.0, 1.0),
            ]
        })
        .collect()
}

/// A cloud with atoms at the range boundaries, as produced by clipping highlights/shadows.
fn clipped_cloud(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 32) as f32 / u32::MAX as f32
    };
    let g = |next: &mut dyn FnMut() -> f32| (next() + next() + next() + next() - 2.0) * 1.5;
    (0..n)
        .map(|_| {
            [
                (0.3 + 0.28 * g(&mut next)).clamp(0.0, 1.0),
                (0.3 + 0.28 * g(&mut next)).clamp(0.0, 1.0),
                (0.3 + 0.28 * g(&mut next)).clamp(0.0, 1.0),
            ]
        })
        .collect()
}

#[test]
fn identity_floor_with_boundary_atoms() {
    let c = clipped_cloud(20_000, 0x1234_5678);
    for k in [1usize, 4, 16, 32] {
        let m = build_model(
            &Samples::new(c.clone(), 1),
            &Samples::new(c.clone(), 1),
            &BuildOptions {
                directions: k,
                ..Default::default()
            },
        )
        .unwrap();
        let mut worst = 0.0f32;
        let mut sum = 0.0f64;
        for p in c.iter().take(3000) {
            let q = apply_point(&m, *p);
            for ch in 0..3 {
                worst = worst.max((q[ch] - p[ch]).abs());
                sum += (q[ch] - p[ch]).abs() as f64;
            }
        }
        println!(
            "clipped K={k:<3} max={:.6} ({:.2} LSB) mean={:.6} ({:.3} LSB)",
            worst,
            worst * 255.0,
            sum / 9000.0,
            sum / 9000.0 * 255.0
        );
    }
}

#[test]
fn identity_floor_by_direction_count() {
    let c = cloud(20_000, 4242);
    let mut prev = f32::INFINITY;
    for k in [1usize, 4, 16, 32] {
        let m = build_model(
            &Samples::new(c.clone(), 1),
            &Samples::new(c.clone(), 1),
            &BuildOptions {
                directions: k,
                ..Default::default()
            },
        )
        .unwrap();
        let mut worst = 0.0f32;
        for p in c.iter().take(3000) {
            let q = apply_point(&m, *p);
            for ch in 0..3 {
                worst = worst.max((q[ch] - p[ch]).abs());
            }
        }
        println!(
            "K={k:<3} max move = {:.6} ({:.2} LSB@8bit)  w1_after = {:.3e}",
            worst,
            worst * 255.0,
            m.stats.w1_after
        );
        assert!(worst < 0.03, "K={k} moved pixels by {worst}");
        prev = prev.min(worst);
    }
    let _ = prev;
}

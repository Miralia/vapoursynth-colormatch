//! The GPU path must agree with the scalar reference.
//!
//! The plugin's contract is that the reported backend is the one that ran. A GPU backend that
//! silently produces different pixels would be worse than having no GPU backend at all, so this
//! compares them directly rather than trusting the transliteration.
//!
//! Tolerance: f32 reassociation and fused multiply-add differ between a CPU and a GPU, so bit
//! equality is not achievable and not the goal. What matters is that the difference is at the level
//! of rounding in the last bits, not at the level of a wrong transform. `3e-5` in normalised RGB is
//! one 16-bit LSB, which is the largest difference the final integer rounding can turn into a
//! one-step output difference; a genuinely wrong transform is off by orders of magnitude more.
//!
//! Skips (rather than fails) when the `gpu` feature is off or no adapter is available, because a
//! machine without a GPU is not a failing test.

#![cfg(feature = "gpu")]

use colormatch_core::apply::{ApplyOptions, Gamut, apply_native};
use colormatch_core::backend::gpu::{apply_gpu, available};
use colormatch_core::build::{BuildOptions, Samples, build_model};
use colormatch_core::model::Model;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as f32 / u32::MAX as f32
    }
}

fn look(x: [f32; 3]) -> [f32; 3] {
    let m = [
        [0.86f32, 0.10, 0.04],
        [0.06, 0.90, 0.04],
        [0.02, 0.12, 0.86],
    ];
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let v = m[c][0] * x[0] + m[c][1] * x[1] + m[c][2] * x[2] + 0.01;
        out[c] = v.clamp(0.0, 1.0).powf(1.05);
    }
    out
}

fn model_for(k: usize, seed: u64) -> Model {
    let mut r = Rng(seed);
    let test: Vec<[f32; 3]> = (0..30_000)
        .map(|_| [r.next(), r.next(), r.next()])
        .collect();
    let reference: Vec<[f32; 3]> = test.iter().copied().map(look).collect();
    build_model(
        &Samples::new(test, 1),
        &Samples::new(reference, 1),
        &BuildOptions {
            directions: k,
            ..Default::default()
        },
    )
    .expect("model builds")
}

/// Returns the worst absolute difference, in normalised RGB.
fn compare(model: &Model, options: &ApplyOptions, seed: u64) -> f32 {
    let mut r = Rng(seed);
    let n = 40_000;
    let (mut cr, mut cg, mut cb): (Vec<f32>, Vec<f32>, Vec<f32>) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        // Deliberately includes values outside [0, 1], where the grid extrapolation and the gamut
        // policy both matter, plus a couple of non-finite samples.
        cr.push(r.next() * 1.6 - 0.3);
        cg.push(r.next() * 1.6 - 0.3);
        cb.push(r.next() * 1.6 - 0.3);
    }
    let (mut gr, mut gg, mut gb) = (cr.clone(), cg.clone(), cb.clone());

    apply_native(model, [&mut cr, &mut cg, &mut cb], options);
    apply_gpu(model, [&mut gr, &mut gg, &mut gb], options, 0).expect("gpu apply");

    let mut worst = 0.0f32;
    for i in 0..n {
        for (c, g) in [(cr[i], gr[i]), (cg[i], gg[i]), (cb[i], gb[i])] {
            if c.is_nan() || g.is_nan() {
                assert!(
                    c.is_nan() == g.is_nan(),
                    "NaN disagreement at {i}: cpu {c} gpu {g}"
                );
                continue;
            }
            worst = worst.max((c - g).abs());
        }
    }
    worst
}

#[test]
fn gpu_matches_scalar_across_models_and_options() {
    if !available(0) {
        eprintln!("skipping: no usable GPU adapter in this environment");
        return;
    }

    // One 16-bit LSB. The gamut and luma-lock steps add their own rounding on top of the slice
    // chain's, so this is the practical floor rather than a generous allowance.
    let tolerance = 3e-5f32;

    for k in [1usize, 4, 16, 32] {
        let model = model_for(k, 0xC0FFEE + k as u64);
        for (label, options) in [
            ("none", ApplyOptions::default()),
            (
                "clip",
                ApplyOptions {
                    gamut: Gamut::Clip,
                    ..Default::default()
                },
            ),
            (
                "soft",
                ApplyOptions {
                    gamut: Gamut::Soft,
                    ..Default::default()
                },
            ),
            (
                "luma_lock",
                ApplyOptions {
                    gamut: Gamut::None,
                    luma_lock: 0.5,
                    ..Default::default()
                },
            ),
        ] {
            let worst = compare(&model, &options, 7 + k as u64);
            assert!(
                worst <= tolerance,
                "K={k} gamut={label}: GPU differs from scalar by {worst:e} (limit {tolerance:e})"
            );
        }
    }
}

#[test]
fn gpu_handles_the_degenerate_inputs_the_same_way() {
    if !available(0) {
        eprintln!("skipping: no usable GPU adapter in this environment");
        return;
    }
    let model = model_for(8, 0x1234);

    // Values that stress the boundary handling: exactly 0 and 1, well outside the grid, and
    // non-finite. A NaN that reaches the tables would poison a whole frame, so the CPU maps it to 0
    // and the GPU must do the same.
    let mut cr = vec![
        0.0f32,
        1.0,
        -5.0,
        5.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.5,
    ];
    let mut cg = cr.clone();
    let mut cb = cr.clone();
    let (mut gr, mut gg, mut gb) = (cr.clone(), cg.clone(), cb.clone());

    let options = ApplyOptions::default();
    apply_native(&model, [&mut cr, &mut cg, &mut cb], &options);
    apply_gpu(&model, [&mut gr, &mut gg, &mut gb], &options, 0).expect("gpu apply");

    for i in 0..cr.len() {
        assert!(
            cr[i].is_finite() && gr[i].is_finite(),
            "index {i}: cpu {} gpu {}",
            cr[i],
            gr[i]
        );
        assert!(
            (cr[i] - gr[i]).abs() <= 3e-5,
            "index {i}: cpu {} gpu {}",
            cr[i],
            gr[i]
        );
    }
}

#[test]
fn gpu_rejects_mismatched_plane_lengths() {
    if !available(0) {
        eprintln!("skipping: no usable GPU adapter in this environment");
        return;
    }
    let model = model_for(4, 9);
    let mut a = vec![0.0f32; 16];
    let mut b = vec![0.0f32; 16];
    let mut c = vec![0.0f32; 8];
    let e = apply_gpu(
        &model,
        [&mut a, &mut b, &mut c],
        &ApplyOptions::default(),
        0,
    )
    .expect_err("mismatched lengths must be rejected");
    assert!(format!("{e}").contains("lengths differ"), "{e}");
}

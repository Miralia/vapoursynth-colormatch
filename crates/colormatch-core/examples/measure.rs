//! Re-measures every number quoted in the README, `docs/algorithm.md` and the crate docs.
//!
//! Run with `cargo run --release -p colormatch-core --example measure`. Keeping the measurements
//! executable means a change that invalidates a documented figure shows up here rather than sitting
//! in prose that nobody re-checks.

use colormatch_core::apply::{ApplyOptions, apply_point};
use colormatch_core::build::{
    Binning, BuildOptions, Builder, CDF_SMOOTH_PASSES, Samples, build_model, sliced_w1,
};
use colormatch_core::lut::{apply_lut_tetrahedral, bake_lut};
use colormatch_core::model::Model;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as f32 / u32::MAX as f32
    }

    /// Sum of four uniforms: bounded, and close enough to normal for this purpose.
    fn gauss(&mut self) -> f32 {
        (0..4).map(|_| self.next()).sum::<f32>() / 2.0 - 1.0
    }
}

/// A cross-channel look plus a per-channel tone curve.
fn apply_look(x: [f32; 3]) -> [f32; 3] {
    let m = [
        [0.86f32, 0.10, 0.04],
        [0.06, 0.90, 0.04],
        [0.02, 0.12, 0.86],
    ];
    let bias = [0.01f32, -0.01, 0.02];
    let gamma = [1.15f32, 1.00, 0.92];
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let v = m[c][0] * x[0] + m[c][1] * x[1] + m[c][2] * x[2] + bias[c];
        out[c] = v.clamp(0.0, 1.0).powf(gamma[c]);
    }
    out
}

fn scene(seed: u64, n: usize, exposure: f32) -> Vec<[f32; 3]> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let mut p = [0.0f32; 3];
            for c in p.iter_mut() {
                *c = ((0.30 + 0.28 * r.gauss()) * exposure).clamp(0.0, 1.0);
            }
            p
        })
        .collect()
}

fn pool(scenes: &[Vec<[f32; 3]>]) -> Vec<[f32; 3]> {
    scenes.iter().flatten().copied().collect()
}

/// Worst error of a 33^3 bake against the exact transform, in 16-bit LSB, over random colours.
fn bake_error_lsb(model: &Model) -> f32 {
    let lut = bake_lut(model, 33).unwrap();
    let mut worst = 0.0f32;
    let mut r = Rng(99);
    for _ in 0..20_000 {
        let p = [r.next(), r.next(), r.next()];
        let exact = apply_point(model, p);
        let mut pr = [p[0]];
        let mut pg = [p[1]];
        let mut pb = [p[2]];
        apply_lut_tetrahedral(
            &lut,
            33,
            [&mut pr, &mut pg, &mut pb],
            &ApplyOptions::default(),
        )
        .unwrap();
        for (e, g) in [exact[0], exact[1], exact[2]]
            .iter()
            .zip([pr[0], pg[0], pb[0]])
        {
            worst = worst.max((e - g).abs() * 65535.0);
        }
    }
    worst
}

fn main() {
    println!("=== 1. Linear splat vs hard binning, cross-channel look, 3 seeds ===");
    for binning in [Binning::Hard, Binning::Splat] {
        let name = match binning {
            Binning::Hard => "hard bins",
            Binning::Splat => "splat    ",
        };
        for k in [32usize, 64] {
            let (mut w1, mut gain, mut bake) = (Vec::new(), Vec::new(), Vec::new());
            for seed in [1u64, 2, 3] {
                let test: Vec<[f32; 3]> = pool(
                    &(0..4)
                        .map(|i| scene(seed * 100 + i, 8_000, 0.75 + 0.5 * i as f32 / 3.0))
                        .collect::<Vec<_>>(),
                );
                let reference: Vec<[f32; 3]> = test.iter().copied().map(apply_look).collect();
                let m = build_model(
                    &Samples::new(test, 4),
                    &Samples::new(reference, 4),
                    &BuildOptions {
                        directions: k,
                        binning,
                        ..Default::default()
                    },
                )
                .unwrap();
                w1.push(m.stats.w1_after);
                gain.push(m.stats.max_gain);
                bake.push(bake_error_lsb(&m));
            }
            println!(
                "  {name} K={k:<3} w1_after {:.5} / {:.5} / {:.5}   gain {:>6.1} / {:>6.1} / {:>6.1}   \
                 bake33 {:>6.1} / {:>6.1} / {:>6.1} LSB",
                w1[0], w1[1], w1[2], gain[0], gain[1], gain[2], bake[0], bake[1], bake[2]
            );
        }
    }

    println!();
    println!("=== 2. How many directions are needed (reduction of pooled sliced-W1) ===");
    let test: Vec<[f32; 3]> = pool(
        &(0..4)
            .map(|i| scene(700 + i, 8_000, 0.75 + 0.5 * i as f32 / 3.0))
            .collect::<Vec<_>>(),
    );
    let reference: Vec<[f32; 3]> = test.iter().copied().map(apply_look).collect();
    for k in [1usize, 2, 4, 8, 10, 16, 32, 64] {
        let m = build_model(
            &Samples::new(test.clone(), 4),
            &Samples::new(reference.clone(), 4),
            &BuildOptions {
                directions: k,
                ..Default::default()
            },
        )
        .unwrap();
        println!(
            "  K={k:<3} w1 {:.5} -> {:.5}  ({:>5.1}% removed)  gain {:.2}",
            m.stats.w1_before,
            m.stats.w1_after,
            (1.0 - m.stats.w1_after / m.stats.w1_before) * 100.0,
            m.stats.max_gain
        );
    }

    println!();
    println!("=== 3. Pooling: one frame vs eight, and held-out scenes ===");
    let build_set: Vec<Vec<[f32; 3]>> = (0..8)
        .map(|i| scene(200 + i, 8_000, 0.75 + 0.5 * i as f32 / 7.0))
        .collect();
    let held_out: Vec<Vec<[f32; 3]>> = (0..4)
        .map(|i| scene(900 + i, 8_000, 0.75 + 0.5 * i as f32 / 3.0))
        .collect();

    let single = build_model(
        &Samples::new(build_set[0].clone(), 1),
        &Samples::new(build_set[0].iter().copied().map(apply_look).collect(), 1),
        &BuildOptions::default(),
    )
    .unwrap();
    let pooled = build_model(
        &Samples::new(pool(&build_set), 8),
        &Samples::new(
            pool(&build_set).iter().copied().map(apply_look).collect(),
            8,
        ),
        &BuildOptions::default(),
    )
    .unwrap();

    for (label, model) in [("frame 0 only", &single), ("8 pooled    ", &pooled)] {
        let evaluate = |frames: &[Vec<[f32; 3]>]| -> (f32, f32) {
            let mut worst = 0.0f32;
            let mut sum = 0.0f32;
            for f in frames {
                let refd: Vec<[f32; 3]> = f.iter().copied().map(apply_look).collect();
                let moved: Vec<[f32; 3]> = f
                    .iter()
                    .map(|&p| {
                        let o = apply_point(model, p);
                        [
                            o[0].clamp(0.0, 1.0),
                            o[1].clamp(0.0, 1.0),
                            o[2].clamp(0.0, 1.0),
                        ]
                    })
                    .collect();
                let d = sliced_w1(&moved, &refd, 16);
                sum += d;
                worst = worst.max(d);
            }
            (sum / frames.len() as f32, worst)
        };
        let (m_build, w_build) = evaluate(&build_set);
        let (m_held, _) = evaluate(&held_out);
        println!(
            "  {label}  build-set mean {m_build:.5} worst {w_build:.5} | held-out mean {m_held:.5}"
        );
    }

    println!();
    println!("=== 4. Sample and bin budget saturation ===");
    for n in [2_000usize, 20_000, 200_000] {
        let mut r = Rng(5);
        let cloud: Vec<[f32; 3]> = (0..n).map(|_| [r.next(), r.next(), r.next()]).collect();
        let refd: Vec<[f32; 3]> = cloud.iter().copied().map(apply_look).collect();
        let m = build_model(
            &Samples::new(cloud, 1),
            &Samples::new(refd, 1),
            &BuildOptions::default(),
        )
        .unwrap();
        println!("  {n:>7} samples  w1_after {:.5}", m.stats.w1_after);
    }
    for bins in [64usize, 128, 256] {
        let mut r = Rng(5);
        let cloud: Vec<[f32; 3]> = (0..40_000)
            .map(|_| [r.next(), r.next(), r.next()])
            .collect();
        let refd: Vec<[f32; 3]> = cloud.iter().copied().map(apply_look).collect();
        let m = build_model(
            &Samples::new(cloud, 1),
            &Samples::new(refd, 1),
            &BuildOptions {
                bins,
                ..Default::default()
            },
        )
        .unwrap();
        println!("  {bins:>7} bins     w1_after {:.5}", m.stats.w1_after);
    }

    println!();
    println!("=== 5. Where the LUT error actually lives ===");
    // A 3D LUT resamples the transform, so its error depends on the transform's local curvature --
    // and on the slope discontinuity where the fitted interior meets the identity tail ramp. Uniform
    // random colours land in the tails often; real content mostly does not. Reporting only one of the
    // two numbers would be misleading in either direction.
    {
        let test: Vec<[f32; 3]> = pool(
            &(0..4)
                .map(|i| scene(700 + i, 8_000, 0.75 + 0.5 * i as f32 / 3.0))
                .collect::<Vec<_>>(),
        );
        let reference: Vec<[f32; 3]> = test.iter().copied().map(apply_look).collect();
        let model = build_model(
            &Samples::new(test.clone(), 4),
            &Samples::new(reference, 4),
            &BuildOptions::default(),
        )
        .unwrap();
        for size in [17u32, 33, 65] {
            let lut = bake_lut(&model, size).unwrap();
            let error = |samples: &[[f32; 3]]| -> (f32, f32) {
                let (mut worst, mut sum) = (0.0f32, 0.0f64);
                for &p in samples {
                    let exact = apply_point(&model, p);
                    let mut pr = [p[0]];
                    let mut pg = [p[1]];
                    let mut pb = [p[2]];
                    apply_lut_tetrahedral(
                        &lut,
                        size,
                        [&mut pr, &mut pg, &mut pb],
                        &ApplyOptions::default(),
                    )
                    .unwrap();
                    for (e, g) in [exact[0], exact[1], exact[2]]
                        .iter()
                        .zip([pr[0], pg[0], pb[0]])
                    {
                        let d = (e - g).abs();
                        worst = worst.max(d);
                        sum += d as f64;
                    }
                }
                (
                    worst * 65535.0,
                    (sum / (samples.len() * 3) as f64) as f32 * 65535.0,
                )
            };
            let (w_in, m_in) = error(&test);
            let mut r = Rng(99);
            let uniform: Vec<[f32; 3]> = (0..20_000)
                .map(|_| [r.next(), r.next(), r.next()])
                .collect();
            let (w_un, m_un) = error(&uniform);
            println!(
                "  {size:>3}^3  in-distribution max {w_in:8.1} mean {m_in:7.2} LSB |                  uniform max {w_un:8.1} mean {m_un:7.2} LSB  (16-bit)"
            );
        }
    }

    println!();
    println!("=== 6. Default builder vs DrDre's structure ===");
    // The docs argue the default is better than the original's approach. This is that argument as a
    // measurement: same clouds, same directions, only the table fit differs.
    {
        let test: Vec<[f32; 3]> = pool(
            &(0..4)
                .map(|i| scene(700 + i, 8_000, 0.75 + 0.5 * i as f32 / 3.0))
                .collect::<Vec<_>>(),
        );
        let reference: Vec<[f32; 3]> = test.iter().copied().map(apply_look).collect();
        for builder in [Builder::DrDre, Builder::Splat] {
            let name = match builder {
                Builder::DrDre => "drdre",
                Builder::Splat => "splat",
            };
            for k in [1usize, 4, 16, 32] {
                let m = build_model(
                    &Samples::new(test.clone(), 4),
                    &Samples::new(reference.clone(), 4),
                    &BuildOptions {
                        directions: k,
                        builder,
                        ..Default::default()
                    },
                )
                .unwrap();
                let monotone = m
                    .slices
                    .iter()
                    .flat_map(|s| s.axes.iter())
                    .all(|a| a.is_monotone(1e-4));
                println!(
                    "  {name} K={k:<3} w1 {:.5} -> {:.5}  ({:>5.1}% removed)  gain {:>7.2}  bake33 {:>8.1} LSB  monotone {monotone}",
                    m.stats.w1_before,
                    m.stats.w1_after,
                    (1.0 - m.stats.w1_after / m.stats.w1_before) * 100.0,
                    m.stats.max_gain,
                    bake_error_lsb(&m),
                );
            }
        }
        // And on a self-match, where the right answer is the identity and any deviation is the
        // builder's own error rather than a mismatch between two distributions.
        for builder in [Builder::DrDre, Builder::Splat] {
            let name = match builder {
                Builder::DrDre => "drdre",
                Builder::Splat => "splat",
            };
            let cloud = scene(42, 40_000, 1.0);
            let m = build_model(
                &Samples::new(cloud.clone(), 1),
                &Samples::new(cloud, 1),
                &BuildOptions {
                    directions: 32,
                    builder,
                    ..Default::default()
                },
            )
            .unwrap();
            println!(
                "  {name} self-match K=32: w1_after {:.2e}  gain {:.3}",
                m.stats.w1_after, m.stats.max_gain
            );
        }
    }

    println!();
    println!("=== 7. CDF smoothing: what it is actually for ===");
    {
        // Chroma upsampling duplicates samples, which puts atoms in the histogram and leaves the
        // empirical CDF stepping. Inverting a stepped CDF is unconstrained, so the map can be
        // arbitrarily steep across the step. Two passes of [1,2,1]/4 turn steps into ramps and bound
        // the slope.
        //
        // The effect is driven by how coarsely the data is quantised, not by duplication: a
        // duplicated continuous cloud is already well conditioned. So this sweeps quantisation
        // levels rather than asserting a single figure.
        let mut rng = Rng(0xA70F);
        println!(
            "  {:<34} {:>10} {:>9} | {:>10} {:>9}",
            "distinct values per channel", "w1 raw", "gain raw", "w1 smooth", "gain sm"
        );
        for distinct in [16usize, 8, 4, 2] {
            let q = |x: f32| {
                ((x * distinct as f32).floor() / (distinct as f32 - 1.0).max(1.0)).clamp(0.0, 1.0)
            };
            let base: Vec<[f32; 3]> = (0..40_000)
                .map(|_| [q(rng.next()), q(rng.next()), q(rng.next())])
                .collect();
            let reference: Vec<[f32; 3]> = base
                .iter()
                .map(|p| {
                    [
                        (p[0] * 0.8 + 0.1).clamp(0.0, 1.0),
                        (p[1] * 0.9 + 0.05).clamp(0.0, 1.0),
                        (p[2] * 0.7 + 0.2).clamp(0.0, 1.0),
                    ]
                })
                .collect();

            let mut row = Vec::new();
            for passes in [0usize, CDF_SMOOTH_PASSES] {
                let m = build_model(
                    &Samples::new(base.clone(), 1),
                    &Samples::new(reference.clone(), 1),
                    &BuildOptions {
                        directions: 8,
                        cdf_smoothing: passes,
                        ..Default::default()
                    },
                )
                .unwrap();
                row.push((m.stats.w1_after, m.stats.max_gain));
            }
            println!(
                "  {:<34} {:>10.5} {:>9.2} | {:>10.5} {:>9.2}",
                distinct, row[0].0, row[0].1, row[1].0, row[1].1
            );
        }
    }

    println!("=== 8. Identity floor: does a self-match stay at the identity? ===");
    for k in [1usize, 4, 16, 32, 64] {
        let cloud = scene(42, 40_000, 1.0);
        let m = build_model(
            &Samples::new(cloud.clone(), 1),
            &Samples::new(cloud, 1),
            &BuildOptions {
                directions: k,
                ..Default::default()
            },
        )
        .unwrap();
        println!(
            "  K={k:<3} w1_after {:.2e}  gain {:.3}",
            m.stats.w1_after, m.stats.max_gain
        );
    }
}

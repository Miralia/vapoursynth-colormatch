//! 3D LUT baking, tetrahedral sampling, and `.cube` / `.clut` export.
//!
//! # Why this is not the default
//!
//! The slice chain is exact; a baked LUT is not. Refining the grid helps far less than the cell size
//! suggests, because the error is driven by *slope* rather than spacing: the transfer function has a
//! kink where the fitted interior meets the identity tail ramp, and a grid cannot resolve a kink.
//!
//! Fine for handing a look to Resolve, and the right fast path at large `K`, but not a substitute for
//! [`crate::apply::apply_native`]. Figures and the rejected smoothing fix are in `docs/algorithm.md`
//! and `examples/measure.rs`.

use crate::apply::ApplyOptions;
use crate::model::Model;

/// Errors from LUT handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LutError {
    /// The requested LUT size is not one of the supported values.
    UnsupportedSize(u32),
    /// The LUT buffer length does not match `size³`.
    BadLutLength {
        /// Declared LUT edge length.
        size: u32,
        /// Actual number of entries supplied.
        len: usize,
    },
    /// The model has no slices, so a LUT would be pointless.
    EmptyModel,
}

impl core::fmt::Display for LutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedSize(s) => write!(f, "unsupported LUT size {s}; use 17, 33 or 65"),
            Self::BadLutLength { size, len } => {
                write!(
                    f,
                    "LUT buffer has {len} entries but size {size} needs {}",
                    size * size * size
                )
            }
            Self::EmptyModel => write!(f, "cannot bake an empty (identity) model to a LUT"),
        }
    }
}

impl std::error::Error for LutError {}

/// LUT sizes this build accepts. Larger grids do not meaningfully reduce the error (see module
/// docs) but do cost cache residency, so the list stays short.
pub const SUPPORTED_SIZES: [u32; 3] = [17, 33, 65];

/// Index of `(r, g, b)` in a LUT whose red channel varies fastest, matching the `.cube` ordering.
#[inline]
pub fn lut_index(r: usize, g: usize, b: usize, size: usize) -> usize {
    (b * size + g) * size + r
}

/// Bakes the model into a 3D LUT of `size³` RGB triples.
///
/// The input grid is the unit cube `[0,1]³`; the output is the transformed colour (unclamped, so
/// callers can decide how to handle out-of-gamut results).
pub fn bake_lut(model: &Model, size: u32) -> Result<Vec<[f32; 3]>, LutError> {
    if !SUPPORTED_SIZES.contains(&size) {
        return Err(LutError::UnsupportedSize(size));
    }
    if model.is_empty() {
        return Err(LutError::EmptyModel);
    }
    let n = size as usize;
    let denom = (size - 1) as f32;
    let mut out = Vec::with_capacity(n * n * n);
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let px = [r as f32 / denom, g as f32 / denom, b as f32 / denom];
                out.push(crate::apply::apply_point(model, px));
            }
        }
    }
    Ok(out)
}

/// Samples a baked LUT with tetrahedral interpolation, in place over three planar `f32` buffers.
///
/// Inputs are clamped to `[0, 1]`, which is the LUT's domain.
pub fn apply_lut_tetrahedral(
    lut: &[[f32; 3]],
    size: u32,
    planes: [&mut [f32]; 3],
    options: &ApplyOptions,
) -> Result<(), LutError> {
    let n = size as usize;
    if lut.len() != n * n * n {
        return Err(LutError::BadLutLength {
            size,
            len: lut.len(),
        });
    }
    let len = planes[0].len();
    debug_assert!(planes[1].len() == len && planes[2].len() == len);
    let denom = (size - 1) as f32;

    // Maps `[0, 1]` to a cell index in `[0, n - 2]` plus an in-cell fraction in `[0, 1]`.
    //
    // Note what this does *not* do: clamp the scaled coordinate to `denom - 1e-6`. In `f32` that
    // subtraction is a no-op once `denom` reaches 64, because the epsilon is below the resolution of
    // the format at that magnitude — `64.0f32 - 1e-6 == 64.0f32`. The index then lands on the last
    // node, and `c(1, 1, 1)` reads one past the end of the LUT: exactly the crash this replaces, and
    // the same trap that bit the histogram splat at `256.0 - 1e-6`. Clamping the *cell* index
    // instead is exact at every size and needs no epsilon.
    let cell = |v: f32| -> (usize, f32) {
        let x = v.clamp(0.0, 1.0) * denom;
        let i0 = (x.floor() as usize).min(n - 2);
        (i0, (x - i0 as f32).clamp(0.0, 1.0))
    };

    let [pr, pg, pb] = planes;
    for i in 0..len {
        let (r0, fr) = cell(pr[i]);
        let (g0, fg) = cell(pg[i]);
        let (b0, fb) = cell(pb[i]);

        let c = |dr: usize, dg: usize, db: usize| lut[lut_index(r0 + dr, g0 + dg, b0 + db, n)];
        let c000 = c(0, 0, 0);
        let c111 = c(1, 1, 1);
        let out = if fr > fg {
            if fg > fb {
                let c100 = c(1, 0, 0);
                let c110 = c(1, 1, 0);
                add3(
                    c000,
                    mul3(fr, sub3(c100, c000)),
                    mul3(fg, sub3(c110, c100)),
                    mul3(fb, sub3(c111, c110)),
                )
            } else if fr > fb {
                let c100 = c(1, 0, 0);
                let c101 = c(1, 0, 1);
                add3(
                    c000,
                    mul3(fr, sub3(c100, c000)),
                    mul3(fb, sub3(c101, c100)),
                    mul3(fg, sub3(c111, c101)),
                )
            } else {
                let c001 = c(0, 0, 1);
                let c101 = c(1, 0, 1);
                add3(
                    c000,
                    mul3(fb, sub3(c001, c000)),
                    mul3(fr, sub3(c101, c001)),
                    mul3(fg, sub3(c111, c101)),
                )
            }
        } else if fb > fg {
            let c001 = c(0, 0, 1);
            let c011 = c(0, 1, 1);
            add3(
                c000,
                mul3(fb, sub3(c001, c000)),
                mul3(fg, sub3(c011, c001)),
                mul3(fr, sub3(c111, c011)),
            )
        } else if fb > fr {
            let c010 = c(0, 1, 0);
            let c011 = c(0, 1, 1);
            add3(
                c000,
                mul3(fg, sub3(c010, c000)),
                mul3(fb, sub3(c011, c010)),
                mul3(fr, sub3(c111, c011)),
            )
        } else {
            let c010 = c(0, 1, 0);
            let c110 = c(1, 1, 0);
            add3(
                c000,
                mul3(fg, sub3(c010, c000)),
                mul3(fr, sub3(c110, c010)),
                mul3(fb, sub3(c111, c110)),
            )
        };

        // The LUT's *domain* is the unit cube, so inputs are clamped on the way in regardless of
        // `gamut` (see the input clamp above). That is an inherent limitation of the LUT path for
        // limited-range YUV content, where legal RGB extends beyond `[0, 1]`; `native` is the
        // default for that reason.
        let (r, g, b) = match options.gamut {
            crate::apply::Gamut::None => (out[0], out[1], out[2]),
            crate::apply::Gamut::Clip | crate::apply::Gamut::Soft => (
                out[0].clamp(0.0, 1.0),
                out[1].clamp(0.0, 1.0),
                out[2].clamp(0.0, 1.0),
            ),
        };
        pr[i] = r;
        pg[i] = g;
        pb[i] = b;
    }
    Ok(())
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn mul3(s: f32, a: [f32; 3]) -> [f32; 3] {
    [s * a[0], s * a[1], s * a[2]]
}
#[inline]
fn add3(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [f32; 3] {
    [
        a[0] + b[0] + c[0] + d[0],
        a[1] + b[1] + c[1] + d[1],
        a[2] + b[2] + c[2] + d[2],
    ]
}

/// Writes the model as an Adobe `.cube` 3D LUT.
///
/// Output is clamped to `[0, 1]` because the format has no way to express out-of-gamut values.
pub fn write_cube(model: &Model, size: u32, title: Option<&str>) -> Result<String, LutError> {
    let lut = bake_lut(model, size)?;
    let mut s = String::with_capacity(lut.len() * 30 + 128);
    if let Some(t) = title {
        s.push_str(&format!("TITLE \"{t}\"\n"));
    }
    s.push_str(&format!("LUT_3D_SIZE {size}\n"));
    s.push_str("DOMAIN_MIN 0.0 0.0 0.0\n");
    s.push_str("DOMAIN_MAX 1.0 1.0 1.0\n\n");
    for px in lut {
        s.push_str(&format!(
            "{:.6} {:.6} {:.6}\n",
            px[0].clamp(0.0, 1.0),
            px[1].clamp(0.0, 1.0),
            px[2].clamp(0.0, 1.0)
        ));
    }
    Ok(s)
}

/// Writes the model as a Resolve `.clut` (the same numeric payload as `.cube`, without a header).
pub fn write_clut(model: &Model, size: u32) -> Result<Vec<u8>, LutError> {
    let lut = bake_lut(model, size)?;
    let mut out = Vec::with_capacity(lut.len() * 12);
    for px in lut {
        for c in px {
            out.extend_from_slice(&c.clamp(0.0, 1.0).to_le_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{BuildOptions, Samples, build_model};
    use crate::model::{Axis, Slice, TABLE_NODES};

    fn simple_model() -> Model {
        // A single slice with an independent per-channel gamma-ish transfer.
        let lo = -1.732_050_8;
        let hi = 1.732_050_8;
        let mut m = Model::identity(TABLE_NODES);
        m.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [
                Axis::from_fn(lo, hi, |x| x.clamp(0.0, 1.0).powf(1.2)),
                Axis::from_fn(lo, hi, |x| x.clamp(0.0, 1.0)),
                Axis::from_fn(lo, hi, |x| x.clamp(0.0, 1.0).powf(0.85)),
            ],
        });
        m
    }

    #[test]
    fn lut_apply_never_indexes_out_of_bounds_at_the_cube_corners() {
        // The corner case that used to crash: an input of exactly 1.0 in all three channels, at every
        // supported size. `denom - 1e-6` does not clamp in f32 once denom reaches 64.
        for size in SUPPORTED_SIZES {
            let n = size as usize;
            let lut: Vec<[f32; 3]> = (0..n * n * n)
                .map(|i| {
                    let (r, g, b) = (i % n, (i / n) % n, i / (n * n));
                    [
                        r as f32 / (n - 1) as f32,
                        g as f32 / (n - 1) as f32,
                        b as f32 / (n - 1) as f32,
                    ]
                })
                .collect();

            for &v in &[0.0f32, 1.0, -0.5, 1.5, 1.0 - f32::EPSILON, 0.5] {
                let mut p0 = vec![v; 4];
                let mut p1 = vec![v; 4];
                let mut p2 = vec![v; 4];
                apply_lut_tetrahedral(
                    &lut,
                    size,
                    [&mut p0, &mut p1, &mut p2],
                    &ApplyOptions::default(),
                )
                .expect("in-range input must not fail");
                // The identity LUT must reproduce its input, including at the corners.
                let expected = v.clamp(0.0, 1.0);
                for out in [p0[0], p1[0], p2[0]] {
                    assert!(
                        (out - expected).abs() < 1e-4,
                        "size {size}, input {v}: expected {expected}, got {out}"
                    );
                }
            }

            // And a full sweep, so no cell boundary is missed.
            let mut p0 = Vec::new();
            let mut p1 = Vec::new();
            let mut p2 = Vec::new();
            for k in 0..=512 {
                let v = k as f32 / 512.0;
                p0.push(v);
                p1.push(1.0 - v);
                p2.push((v * 3.0).fract());
            }
            apply_lut_tetrahedral(
                &lut,
                size,
                [&mut p0, &mut p1, &mut p2],
                &ApplyOptions::default(),
            )
            .expect("sweep must not fail");
        }
    }

    #[test]
    fn lut_index_has_red_varying_fastest() {
        assert_eq!(lut_index(0, 0, 0, 33), 0);
        assert_eq!(lut_index(1, 0, 0, 33), 1);
        assert_eq!(lut_index(0, 1, 0, 33), 33);
        assert_eq!(lut_index(0, 0, 1, 33), 33 * 33);
    }

    #[test]
    fn bake_rejects_bad_sizes_and_empty_models() {
        assert_eq!(
            bake_lut(&simple_model(), 32),
            Err(LutError::UnsupportedSize(32))
        );
        assert_eq!(
            bake_lut(&Model::identity(TABLE_NODES), 33),
            Err(LutError::EmptyModel)
        );
    }

    #[test]
    fn baked_lut_is_exact_on_grid_nodes() {
        let m = simple_model();
        let lut = bake_lut(&m, 33).unwrap();
        let denom = 32.0f32;
        for b in 0..33usize {
            for g in 0..33usize {
                for r in 0..33usize {
                    let px = [r as f32 / denom, g as f32 / denom, b as f32 / denom];
                    let want = crate::apply::apply_point(&m, px);
                    let got = lut[lut_index(r, g, b, 33)];
                    for c in 0..3 {
                        assert!(
                            (want[c] - got[c]).abs() < 1e-5,
                            "at {px:?} ch{c}: {want:?} vs {got:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tetrahedral_interpolation_is_exact_for_an_affine_transform() {
        // A linear map is reproduced exactly by tetrahedral interpolation.
        let mut m = Model::identity(TABLE_NODES);
        let lo = -1.732_050_8;
        let hi = 1.732_050_8;
        // Diagonal gain only: representable as identity axes with a slope.
        m.slices.push(Slice {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [
                Axis::from_fn(lo, hi, |x| x * 0.5 + 0.25),
                Axis::from_fn(lo, hi, |x| x * 0.25 + 0.5),
                Axis::from_fn(lo, hi, |x| x * 0.75),
            ],
        });
        let lut = bake_lut(&m, 33).unwrap();
        let n = 1000usize;
        let mut r: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let mut g: Vec<f32> = (0..n).map(|i| ((i * 7) % n) as f32 / n as f32).collect();
        let mut b: Vec<f32> = (0..n).map(|i| ((i * 13) % n) as f32 / n as f32).collect();
        let want: Vec<[f32; 3]> = (0..n)
            .map(|i| crate::apply::apply_point(&m, [r[i], g[i], b[i]]))
            .collect();
        apply_lut_tetrahedral(&lut, 33, [&mut r, &mut g, &mut b], &ApplyOptions::default())
            .unwrap();
        let mut worst = 0.0f32;
        for i in 0..n {
            for c in 0..3 {
                worst = worst.max(([r[i], g[i], b[i]][c] - want[i][c]).abs());
            }
        }
        assert!(
            worst < 1e-5,
            "affine map not reproduced exactly, worst={worst}"
        );
    }

    #[test]
    fn lut_length_is_validated() {
        let lut = vec![[0.0f32; 3]; 10];
        let (mut r, mut g, mut b) = (vec![0.0f32; 4], vec![0.0f32; 4], vec![0.0f32; 4]);
        assert_eq!(
            apply_lut_tetrahedral(&lut, 33, [&mut r, &mut g, &mut b], &ApplyOptions::default()),
            Err(LutError::BadLutLength { size: 33, len: 10 })
        );
    }

    #[test]
    fn baked_lut_approximates_a_real_model_within_a_few_lsb() {
        // Build a real cross-channel model, then compare the baked path with the exact path.
        let mut s = 0x9E37_79B9u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as f32 / u32::MAX as f32
        };
        let n = 20_000;
        let test: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    (0.2 + 0.6 * next()).clamp(0.0, 1.0),
                    (0.2 + 0.6 * next()).clamp(0.0, 1.0),
                    (0.2 + 0.6 * next()).clamp(0.0, 1.0),
                ]
            })
            .collect();
        let m = [
            [0.86f32, 0.10, 0.04],
            [0.06, 0.90, 0.04],
            [0.02, 0.12, 0.86],
        ];
        let reference: Vec<[f32; 3]> = test
            .iter()
            .map(|p| {
                let mut y = [0.0f32; 3];
                for i in 0..3 {
                    y[i] = (m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + 0.01)
                        .clamp(0.0, 1.0)
                        .powf(1.05);
                }
                y
            })
            .collect();

        let model = build_model(
            &Samples::new(test.clone(), 8),
            &Samples::new(reference, 8),
            &BuildOptions {
                directions: 32,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(model.stats.w1_after < model.stats.w1_before * 0.05);

        let lut = bake_lut(&model, 33).unwrap();
        let mut r: Vec<f32> = test.iter().map(|p| p[0]).collect();
        let mut g: Vec<f32> = test.iter().map(|p| p[1]).collect();
        let mut b: Vec<f32> = test.iter().map(|p| p[2]).collect();
        let exact: Vec<[f32; 3]> = test
            .iter()
            .map(|p| crate::apply::apply_point(&model, *p))
            .collect();
        apply_lut_tetrahedral(&lut, 33, [&mut r, &mut g, &mut b], &ApplyOptions::default())
            .unwrap();

        let mut worst = 0.0f32;
        let mut sum = 0.0f64;
        for i in 0..test.len() {
            let got = [r[i], g[i], b[i]];
            for c in 0..3 {
                let d = (got[c] - exact[i][c].clamp(0.0, 1.0)).abs();
                worst = worst.max(d);
                sum += d as f64;
            }
        }
        let mean = sum / (test.len() * 3) as f64;
        assert!(
            worst * 255.0 < 16.0,
            "33^3 bake worst error {:.1} LSB",
            worst * 255.0
        );
        assert!(
            mean * 255.0 < 2.0,
            "33^3 bake mean error {:.2} LSB",
            mean * 255.0
        );
    }

    #[test]
    fn cube_export_has_a_parseable_header_and_the_right_row_count() {
        let text = write_cube(&simple_model(), 17, Some("unit test")).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("TITLE \"unit test\""));
        assert_eq!(lines.next(), Some("LUT_3D_SIZE 17"));
        assert_eq!(lines.next(), Some("DOMAIN_MIN 0.0 0.0 0.0"));
        assert_eq!(lines.next(), Some("DOMAIN_MAX 1.0 1.0 1.0"));
        let rows: Vec<&str> = text
            .lines()
            .filter(|l| {
                !l.is_empty()
                    && !l.starts_with("TITLE")
                    && !l.starts_with("LUT_3D_SIZE")
                    && !l.starts_with("DOMAIN")
            })
            .collect();
        assert_eq!(rows.len(), 17 * 17 * 17);
        for row in rows.iter().take(5) {
            let parts: Vec<f32> = row.split_whitespace().map(|p| p.parse().unwrap()).collect();
            assert_eq!(parts.len(), 3);
            assert!(parts.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        // First entry is the black point of the transform, last is the white point.
        let first: Vec<f32> = rows[0]
            .split_whitespace()
            .map(|p| p.parse().unwrap())
            .collect();
        assert!(
            first.iter().all(|v| *v < 0.01),
            "black point should stay black: {first:?}"
        );
    }

    #[test]
    fn clut_export_has_the_expected_byte_length() {
        let bytes = write_clut(&simple_model(), 17).unwrap();
        assert_eq!(bytes.len(), 17 * 17 * 17 * 3 * 4);
    }
}

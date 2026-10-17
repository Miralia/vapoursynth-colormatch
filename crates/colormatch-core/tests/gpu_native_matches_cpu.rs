//! The GPU native-sample path against the CPU reference.
//!
//! `apply_gpu_native` moves the format conversion, the YUV matrix and the chroma resampling onto the
//! GPU. That is only acceptable if it agrees with the CPU path, so this compares them on synthetic
//! frames in every layout the path supports, and requires an identity model to be an *exact* no-op —
//! the property that has found four real defects in this codebase.
//!
//! The CPU reference here is written out longhand rather than called from the plugin crate, because
//! the plugin's `frame_to_rgb`/`rgb_to_frame` need VapourSynth frame types and this test must run
//! without them.

#![cfg(feature = "gpu")]

use colormatch_core::apply::{ApplyOptions, apply_native};
use colormatch_core::backend::gpu::{
    NativeFamily, NativeLayout, NativeSample, apply_gpu_native, available,
};
use colormatch_core::build::{BuildOptions, Samples, build_model};
use colormatch_core::color::{
    PlaneKind, SpaceInfo, denormalize_sample, luma_coefficients, normalize_sample, plane_scale,
    rgb_to_yuv, yuv_to_rgb,
};
use colormatch_core::model::Model;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
    }
}

/// One layout under test.
struct Case {
    name: &'static str,
    family: NativeFamily,
    width: u32,
    height: u32,
    sub_h: u32,
    sub_v: u32,
    matrix: u8,
    full_range: bool,
}

impl Case {
    fn cw(&self) -> u32 {
        self.width / self.sub_h
    }
    fn ch(&self) -> u32 {
        self.height / self.sub_v
    }
    fn bits(&self) -> u32 {
        16
    }
    fn luma_scale(&self) -> colormatch_core::color::PlaneScale {
        plane_scale(self.bits(), self.full_range, PlaneKind::Luma)
    }
    fn chroma_scale(&self) -> colormatch_core::color::PlaneScale {
        plane_scale(self.bits(), self.full_range, PlaneKind::Chroma)
    }
    fn layout(&self) -> NativeLayout {
        let (kr, kb) = luma_coefficients(self.matrix).unwrap_or((0.2126, 0.0722));
        NativeLayout {
            sample: NativeSample::U16,
            family: self.family,
            width: self.width,
            height: self.height,
            cw: self.cw(),
            ch: self.ch(),
            sub_h: self.sub_h,
            sub_v: self.sub_v,
            luma: self.luma_scale(),
            chroma: self.chroma_scale(),
            kr,
            kb,
        }
    }
}

/// Bilinear chroma upsample onto the luma grid, mirroring `upsample_plane`.
fn upsample(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; dw * dh];
    let (sx, sy) = (sw as f32 / dw as f32, sh as f32 / dh as f32);
    for y in 0..dh {
        let fy = y as f32 * sy;
        let y0f = fy.floor().min(sh as f32 - 1.0);
        let y1f = (y0f + 1.0).min(sh as f32 - 1.0);
        let ty = (fy - y0f).clamp(0.0, 1.0);
        for x in 0..dw {
            let fx = x as f32 * sx;
            let x0f = fx.floor().min(sw as f32 - 1.0);
            let x1f = (x0f + 1.0).min(sw as f32 - 1.0);
            let tx = (fx - x0f).clamp(0.0, 1.0);
            let (x0, x1) = (x0f as usize, x1f as usize);
            let (y0, y1) = (y0f as usize, y1f as usize);
            let top = src[y0 * sw + x0] + (src[y0 * sw + x1] - src[y0 * sw + x0]) * tx;
            let bot = src[y1 * sw + x0] + (src[y1 * sw + x1] - src[y1 * sw + x0]) * tx;
            out[y * dw + x] = top + (bot - top) * ty;
        }
    }
    out
}

/// The CPU reference for one case: native u16 planes in, native u16 planes out.
fn cpu_reference(
    case: &Case,
    model: &Model,
    options: &ApplyOptions,
    planes: &[Vec<u16>; 3],
) -> [Vec<u16>; 3] {
    let bits = case.bits();
    let full = case.full_range;
    let (w, h) = (case.width as usize, case.height as usize);
    let (cw, ch) = (case.cw() as usize, case.ch() as usize);
    let n = w * h;

    // Normalise, and upsample chroma when subsampled.
    let lum = case.luma_scale();
    let chr = case.chroma_scale();
    let norm = |raw: f32, s: &colormatch_core::color::PlaneScale| {
        (raw * s.norm_scale + s.norm_offset).clamp(s.norm_lo, s.norm_hi)
    };
    let y: Vec<f32> = planes[0][..n]
        .iter()
        .map(|&v| norm(v as f32, &lum))
        .collect();
    let cu: Vec<f32> = planes[1][..cw * ch]
        .iter()
        .map(|&v| norm(v as f32, &chr))
        .collect();
    let cv: Vec<f32> = planes[2][..cw * ch]
        .iter()
        .map(|&v| norm(v as f32, &chr))
        .collect();
    let u = upsample(&cu, cw, ch, w, h);
    let v = upsample(&cv, cw, ch, w, h);

    // To RGB, transform, back.
    let (kr, kb) = luma_coefficients(case.matrix).unwrap_or((0.2126, 0.0722));
    let (mut pr, mut pg, mut pb) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    match case.family {
        NativeFamily::Gray => {
            pr.copy_from_slice(&y);
            pg.copy_from_slice(&y);
            pb.copy_from_slice(&y);
        }
        NativeFamily::Rgb => {
            pr.copy_from_slice(
                &planes[0][..n]
                    .iter()
                    .map(|&x| norm(x as f32, &lum))
                    .collect::<Vec<_>>(),
            );
            pg.copy_from_slice(
                &planes[1][..n]
                    .iter()
                    .map(|&x| norm(x as f32, &lum))
                    .collect::<Vec<_>>(),
            );
            pb.copy_from_slice(
                &planes[2][..n]
                    .iter()
                    .map(|&x| norm(x as f32, &lum))
                    .collect::<Vec<_>>(),
            );
        }
        NativeFamily::Yuv => {
            for i in 0..n {
                let rgb = yuv_to_rgb(y[i], u[i], v[i], kr, kb);
                pr[i] = rgb[0];
                pg[i] = rgb[1];
                pb[i] = rgb[2];
            }
        }
    }
    apply_native(model, [&mut pr, &mut pg, &mut pb], options);

    // Back to native.
    let denorm = |val: f32, s: &colormatch_core::color::PlaneScale| -> u16 {
        (val * s.denorm_scale + s.denorm_offset)
            .clamp(0.0, s.peak)
            .round() as u16
    };
    let mut out0 = vec![0u16; n];
    match case.family {
        NativeFamily::Gray => {
            let kg = 1.0 - kr - kb;
            for i in 0..n {
                out0[i] = denorm(kr * pr[i] + kg * pg[i] + kb * pb[i], &lum);
            }
            return [out0, vec![0u16; cw * ch], vec![0u16; cw * ch]];
        }
        NativeFamily::Rgb => {
            for i in 0..n {
                out0[i] = denorm(pr[i], &lum);
            }
            let o1: Vec<u16> = (0..n).map(|i| denorm(pg[i], &lum)).collect();
            let o2: Vec<u16> = (0..n).map(|i| denorm(pb[i], &lum)).collect();
            return [out0, o1, o2];
        }
        NativeFamily::Yuv => {}
    }

    let mut fu = vec![0.0f32; n];
    let mut fv = vec![0.0f32; n];
    for i in 0..n {
        let t = rgb_to_yuv(pr[i], pg[i], pb[i], kr, kb);
        out0[i] = denorm(t[0], &lum);
        fu[i] = t[1];
        fv[i] = t[2];
    }
    // Nearest downsample: chroma `(i, j)` is co-sited with luma `(i * sub_h, j * sub_v)`.
    let mut o1 = vec![0u16; cw * ch];
    let mut o2 = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let si = (j * case.sub_v as usize) * w + i * case.sub_h as usize;
            o1[j * cw + i] = denorm(fu[si], &chr);
            o2[j * cw + i] = denorm(fv[si], &chr);
        }
    }
    let _ = bits;
    let _ = full;
    let _ = SpaceInfo::WORKING;
    let _ = normalize_sample as fn(f32, u32, bool, PlaneKind) -> f32;
    let _ = denormalize_sample as fn(f32, u32, bool, PlaneKind) -> f32;
    [out0, o1, o2]
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn from_u16_bytes(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn gpu_native_matches_the_cpu_reference() {
    if !available(0) {
        eprintln!("no usable GPU adapter; skipping");
        return;
    }
    let cases = [
        Case {
            name: "rgb full",
            family: NativeFamily::Rgb,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 0,
            full_range: true,
        },
        Case {
            name: "rgb limited",
            family: NativeFamily::Rgb,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 0,
            full_range: false,
        },
        Case {
            name: "gray full",
            family: NativeFamily::Gray,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 1,
            full_range: true,
        },
        Case {
            name: "yuv444 limited",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 1,
            full_range: false,
        },
        Case {
            name: "yuv444 full",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 9,
            full_range: true,
        },
        Case {
            name: "yuv422 limited",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 2,
            sub_v: 1,
            matrix: 1,
            full_range: false,
        },
        Case {
            name: "yuv420 limited",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 2,
            sub_v: 2,
            matrix: 1,
            full_range: false,
        },
        Case {
            name: "yuv420 full",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 2,
            sub_v: 2,
            matrix: 9,
            full_range: true,
        },
    ];

    let mut rng = Rng(0x9E3779B9);
    let cloud_a: Vec<[f32; 3]> = (0..30_000)
        .map(|_| [rng.next(), rng.next(), rng.next()])
        .collect();
    let cloud_b: Vec<[f32; 3]> = cloud_a
        .iter()
        .map(|p| {
            [
                (p[0] * 0.8 + 0.1).clamp(0.0, 1.0),
                (p[1] * 0.9 + 0.05).clamp(0.0, 1.0),
                (p[2] * 0.7 + 0.2).clamp(0.0, 1.0),
            ]
        })
        .collect();
    let model = build_model(
        &Samples::new(cloud_a, 1),
        &Samples::new(cloud_b, 1),
        &BuildOptions {
            directions: 16,
            ..Default::default()
        },
    )
    .unwrap();
    let options = ApplyOptions::default();

    for case in &cases {
        let (w, h) = (case.width as usize, case.height as usize);
        let (cw, ch) = (case.cw() as usize, case.ch() as usize);
        let n = w * h;
        let count = if case.family == NativeFamily::Gray {
            1
        } else {
            3
        };
        let len = |p: usize| {
            if case.family == NativeFamily::Yuv && p > 0 {
                cw * ch
            } else {
                n
            }
        };

        let planes: [Vec<u16>; 3] =
            [0, 1, 2].map(|p| (0..len(p)).map(|_| (rng.next() * 65535.0) as u16).collect());

        let expected = cpu_reference(case, &model, &options, &planes);

        let in_bytes: Vec<Vec<u8>> = (0..3).map(|p| u16_bytes(&planes[p])).collect();
        // Distinct bindings rather than a `Vec` of slices: `&mut out[p]` cannot borrow-disjoint.
        let mut out_bytes: [Vec<u8>; 3] = [0, 1, 2].map(|p| vec![0u8; len(p) * 2]);
        let [o0, o1, o2] = &mut out_bytes;
        apply_gpu_native(
            &model,
            [
                (in_bytes[0].as_slice(), o0.as_mut_slice()),
                (in_bytes[1].as_slice(), o1.as_mut_slice()),
                (in_bytes[2].as_slice(), o2.as_mut_slice()),
            ],
            &case.layout(),
            &options,
            0,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));

        for p in 0..count {
            let got = from_u16_bytes(&out_bytes[p]);
            let want = &expected[p];
            let worst = got
                .iter()
                .zip(want)
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);
            assert!(
                worst <= 2,
                "{} plane {p}: max difference {worst} of 65535 exceeds 2 LSB",
                case.name
            );
        }
    }
}

/// The identity property: an exact no-op, which is what lets this backend be swapped for the CPU one.
///
/// Uses a *truly* identity model — every table is the grid itself — because a self-match build is
/// only approximately identity and so cannot distinguish "the conversion is inexact" from "the model
/// has a residual the GPU rounds differently".
#[test]
fn gpu_native_identity_is_an_exact_noop() {
    if !available(0) {
        eprintln!("no usable GPU adapter; skipping");
        return;
    }
    let cases = [
        Case {
            name: "rgb full",
            family: NativeFamily::Rgb,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 0,
            full_range: true,
        },
        Case {
            name: "rgb limited",
            family: NativeFamily::Rgb,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 0,
            full_range: false,
        },
        Case {
            name: "yuv444 limited",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 1,
            sub_v: 1,
            matrix: 1,
            full_range: false,
        },
        Case {
            name: "yuv422 limited",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 2,
            sub_v: 1,
            matrix: 1,
            full_range: false,
        },
        Case {
            name: "yuv420 limited",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 2,
            sub_v: 2,
            matrix: 1,
            full_range: false,
        },
        Case {
            name: "yuv420 full",
            family: NativeFamily::Yuv,
            width: 64,
            height: 48,
            sub_h: 2,
            sub_v: 2,
            matrix: 9,
            full_range: true,
        },
    ];

    let mut rng = Rng(0x1234_5678);
    let cloud: Vec<[f32; 3]> = (0..30_000)
        .map(|_| [rng.next(), rng.next(), rng.next()])
        .collect();
    // A *truly* identity model: every table is the grid itself. A self-match build is only
    // approximately identity, so it cannot distinguish "the conversion is inexact" from "the model
    // has a residual that the GPU rounds differently".
    let mut identity = build_model(
        &Samples::new(cloud.clone(), 1),
        &Samples::new(cloud, 1),
        &BuildOptions {
            directions: 16,
            ..Default::default()
        },
    )
    .unwrap();
    for slice in &mut identity.slices {
        for axis in &mut slice.axes {
            // The grid node for index `j` is `lo + (hi - lo) * j / (n - 1)`, and `scale` is
            // `(n - 1) / (hi - lo)`, so the identity table is simply `lo + j / scale`.
            let (lo, scale) = (axis.lo, axis.scale);
            for (j, node) in axis.tab.iter_mut().enumerate() {
                *node = lo + j as f32 / scale;
            }
        }
    }
    let options = ApplyOptions::default();

    for case in &cases {
        let (w, h) = (case.width as usize, case.height as usize);
        let (cw, ch) = (case.cw() as usize, case.ch() as usize);
        let n = w * h;
        let len = |p: usize| {
            if case.family == NativeFamily::Yuv && p > 0 {
                cw * ch
            } else {
                n
            }
        };

        // The input has to be a *legal* frame, not independently random planes. Random luma with
        // random chroma is not a colour that exists: it reconstructs to RGB well outside the cube,
        // which the input clamp then pulls back, and the round trip is legitimately not exact. Real
        // content — and the integration suite — derives YUV from RGB, so this does too.
        let lum_s = case.luma_scale();
        let chr_s = case.chroma_scale();
        let quant = |v: f32, sc: &colormatch_core::color::PlaneScale| -> u16 {
            (v * sc.denorm_scale + sc.denorm_offset)
                .clamp(0.0, sc.peak)
                .round() as u16
        };
        let (kr, kb) = luma_coefficients(case.matrix).unwrap_or((0.2126, 0.0722));
        let planes: [Vec<u16>; 3] = match case.family {
            NativeFamily::Rgb => {
                [0, 1, 2].map(|_| (0..n).map(|_| (rng.next() * 65535.0) as u16).collect())
            }
            NativeFamily::Gray => {
                let y: Vec<u16> = (0..n).map(|_| (rng.next() * 65535.0) as u16).collect();
                [y.clone(), y.clone(), y]
            }
            NativeFamily::Yuv => {
                let mut y = vec![0u16; n];
                let mut cu = vec![0u16; cw * ch];
                let mut cv = vec![0u16; cw * ch];
                // One random RGB pixel per output chroma position, so luma and chroma agree.
                for j in 0..ch {
                    for i in 0..cw {
                        let (r, g, b) = (rng.next(), rng.next(), rng.next());
                        let t = rgb_to_yuv(r, g, b, kr, kb);
                        cu[j * cw + i] = quant(t[1], &chr_s);
                        cv[j * cw + i] = quant(t[2], &chr_s);
                        for dy in 0..case.sub_v as usize {
                            for dx in 0..case.sub_h as usize {
                                let (x, yy) =
                                    (i * case.sub_h as usize + dx, j * case.sub_v as usize + dy);
                                if x < w && yy < h {
                                    y[yy * w + x] = quant(t[0], &lum_s);
                                }
                            }
                        }
                    }
                }
                [y, cu, cv]
            }
        };

        let in_bytes: Vec<Vec<u8>> = (0..3).map(|p| u16_bytes(&planes[p])).collect();
        let mut out_bytes: [Vec<u8>; 3] = [0, 1, 2].map(|p| vec![0u8; len(p) * 2]);
        let [o0, o1, o2] = &mut out_bytes;
        apply_gpu_native(
            &identity,
            [
                (in_bytes[0].as_slice(), o0.as_mut_slice()),
                (in_bytes[1].as_slice(), o1.as_mut_slice()),
                (in_bytes[2].as_slice(), o2.as_mut_slice()),
            ],
            &case.layout(),
            &options,
            0,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));

        for p in 0..3 {
            let got = from_u16_bytes(&out_bytes[p]);
            let worst = got
                .iter()
                .zip(&planes[p])
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);
            assert_eq!(
                worst, 0,
                "{} plane {p}: an identity model must be an exact no-op, worst {worst}",
                case.name
            );
        }
    }
}

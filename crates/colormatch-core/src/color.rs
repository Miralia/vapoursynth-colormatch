//! Colour space description and YUV ↔ RGB conversion.
//!
//! The plugin converts every input into a single internal RGB working space before either
//! sampling (build) or transforming (apply). Using the *same* conversion on both paths is a
//! correctness requirement: if the model is built in one space and applied in another, the match
//! silently degrades instead of failing.
//!
//! Codes match VapourSynth's `VSMatrixCoefficients`, `VSColorRange`, `VSTransferCharacteristics`
//! and `VSColorPrimaries` enums so that frame properties can be copied verbatim.

/// VapourSynth `VSColorRange`.
pub mod range {
    /// Studio / limited range (16–235 luma for 8-bit).
    pub const LIMITED: u8 = 0;
    /// Full range (0–255 luma for 8-bit).
    pub const FULL: u8 = 1;
}

/// VapourSynth `VSMatrixCoefficients`.
pub mod matrix {
    /// Identity: the input is already RGB.
    pub const RGB: u8 = 0;
    /// Rec. ITU-R BT.709.
    pub const BT709: u8 = 1;
    /// Unspecified; callers must resolve this before it reaches the core.
    pub const UNSPECIFIED: u8 = 2;
    /// FCC.
    pub const FCC: u8 = 4;
    /// Rec. ITU-R BT.470BG.
    pub const BT470BG: u8 = 5;
    /// SMPTE 170M.
    pub const SMPTE170M: u8 = 6;
    /// SMPTE 240M.
    pub const SMPTE240M: u8 = 7;
    /// YCgCo.
    pub const YCGCO: u8 = 8;
    /// Rec. ITU-R BT.2020 non-constant luminance.
    pub const BT2020NC: u8 = 9;
    /// Rec. ITU-R BT.2020 constant luminance.
    pub const BT2020C: u8 = 10;
}

/// VapourSynth `VSColorPrimaries`.
pub mod primaries {
    /// BT.709.
    pub const BT709: u8 = 1;
    /// Unspecified.
    pub const UNSPECIFIED: u8 = 2;
    /// BT.470M.
    pub const BT470M: u8 = 4;
    /// BT.470BG.
    pub const BT470BG: u8 = 5;
    /// SMPTE 170M.
    pub const SMPTE170M: u8 = 6;
    /// SMPTE 240M.
    pub const SMPTE240M: u8 = 7;
    /// Generic film.
    pub const FILM: u8 = 8;
    /// BT.2020.
    pub const BT2020: u8 = 9;
}

/// VapourSynth `VSTransferCharacteristics`.
pub mod transfer {
    /// Unspecified.
    pub const UNSPECIFIED: u8 = 2;
    /// BT.709 / BT.601 / SMPTE 170M.
    pub const BT709: u8 = 1;
    /// sRGB (piecewise, 12.92 / 1.055).
    pub const SRGB: u8 = 13;
    /// Linear light.
    pub const LINEAR: u8 = 8;
    /// SMPTE 2084 (PQ).
    pub const ST2084: u8 = 16;
    /// ARIB STD-B67 (HLG).
    pub const ARIB_STD_B67: u8 = 18;
}

/// The colour space a model was built for, or that an input is being interpreted as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpaceInfo {
    /// `VSMatrixCoefficients`.
    pub matrix: u8,
    /// `VSColorRange`.
    pub range: u8,
    /// `VSTransferCharacteristics`.
    pub transfer: u8,
    /// `VSColorPrimaries`.
    pub primaries: u8,
}

impl SpaceInfo {
    /// A full-range RGB space with BT.709 transfer and primaries — the internal working space.
    pub const WORKING: Self = Self {
        matrix: matrix::RGB,
        range: range::FULL,
        transfer: transfer::BT709,
        primaries: primaries::BT709,
    };

    /// Packs into four bytes.
    pub fn to_bytes(self) -> [u8; 4] {
        [self.matrix, self.range, self.transfer, self.primaries]
    }

    /// Unpacks from four bytes. Out-of-range codes are preserved as-is; validation happens in
    /// [`Self::validate`].
    pub fn from_bytes(b: &[u8]) -> Self {
        Self {
            matrix: b.first().copied().unwrap_or(0),
            range: b.get(1).copied().unwrap_or(0),
            transfer: b.get(2).copied().unwrap_or(0),
            primaries: b.get(3).copied().unwrap_or(0),
        }
    }

    /// Returns `true` when this space is RGB (no matrix conversion needed).
    pub fn is_rgb(self) -> bool {
        self.matrix == matrix::RGB
    }

    /// Rejects spaces this build cannot interpret, instead of guessing.
    pub fn validate(self) -> Result<(), ColorError> {
        if self.is_rgb() {
            return Ok(());
        }
        luma_coefficients(self.matrix)?;
        Ok(())
    }
}

/// Errors from colour space handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorError {
    /// The matrix coefficient code is not supported (including `UNSPECIFIED`).
    UnsupportedMatrix(u8),
    /// Constant-luminance BT.2020 is not supported.
    ConstantLuminanceUnsupported,
    /// The transfer characteristic code is not supported for linearisation.
    UnsupportedTransfer(u8),
    /// The bit depth is not one of 8/10/12/14/16/32.
    UnsupportedDepth(u32),
}

impl core::fmt::Display for ColorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedMatrix(m) => write!(
                f,
                "unsupported matrix coefficients {m}; set an explicit `matrix` or tag the clip"
            ),
            Self::ConstantLuminanceUnsupported => {
                write!(f, "BT.2020 constant-luminance (CL) is not supported")
            }
            Self::UnsupportedTransfer(t) => write!(f, "unsupported transfer characteristics {t}"),
            Self::UnsupportedDepth(d) => write!(f, "unsupported bit depth {d}"),
        }
    }
}

impl std::error::Error for ColorError {}

/// Kr/Kb luma coefficients for a matrix coefficient code.
///
/// Returns `(kr, kb)`. `BT.2020 CL` is rejected rather than silently treated as NCL.
pub fn luma_coefficients(m: u8) -> Result<(f32, f32), ColorError> {
    match m {
        matrix::BT709 => Ok((0.2126, 0.0722)),
        matrix::FCC => Ok((0.30, 0.11)),
        matrix::BT470BG | matrix::SMPTE170M => Ok((0.299, 0.114)),
        matrix::SMPTE240M => Ok((0.212, 0.087)),
        matrix::BT2020NC => Ok((0.2627, 0.0593)),
        matrix::YCGCO => Ok((0.25, 0.25)),
        matrix::BT2020C => Err(ColorError::ConstantLuminanceUnsupported),
        other => Err(ColorError::UnsupportedMatrix(other)),
    }
}

/// Which plane a sample belongs to.
///
/// Luma and chroma use *different* legal ranges, in both limited and full range, and chroma is
/// **centred on zero** rather than on 0.5. Treating the two the same is a silent, large error: at
/// 10-bit limited range, neutral chroma (128·4 = 512) would normalise to 0.5 instead of 0.0, so
/// every YUV→RGB conversion would be offset by half the chroma range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneKind {
    /// The luma plane (Y).
    Luma,
    /// A chroma plane (Cb/Cr, U/V).
    Chroma,
}

/// Normalises an integer sample to the working range.
///
/// * Luma, full range: `[0, peak]` → `[0, 1]`
/// * Luma, limited range: `[16s, 235s]` → `[0, 1]`
/// * Chroma, full range: `[0, peak]` → `[-0.5, 0.5]`
/// * Chroma, limited range: `[128s ± 112s]` → `[-0.5, 0.5]`
///
/// where `s = 2^(bits - 8)` scales the 8-bit studio-swing constants to the sample depth.
///
/// The result is *not* clamped to the legal range: undershoot and overshoot are legal in video and
/// carry information. The bounds here only keep a pathological input from producing an absurd value.
#[inline]
pub fn normalize_sample(raw: f32, bits: u32, full_range: bool, kind: PlaneKind) -> f32 {
    let peak = ((1u64 << bits) - 1) as f32;
    match (full_range, kind) {
        (true, PlaneKind::Luma) => raw / peak,
        (true, PlaneKind::Chroma) => (raw / peak - 0.5).clamp(-1.0, 1.0),
        (false, PlaneKind::Luma) => {
            let scale = (1u32 << (bits - 8)) as f32;
            ((raw - 16.0 * scale) / (219.0 * scale)).clamp(-0.5, 1.5)
        }
        (false, PlaneKind::Chroma) => {
            let scale = (1u32 << (bits - 8)) as f32;
            ((raw - 128.0 * scale) / (224.0 * scale)).clamp(-1.0, 1.0)
        }
    }
}

/// Inverse of [`normalize_sample`].
#[inline]
pub fn denormalize_sample(v: f32, bits: u32, full_range: bool, kind: PlaneKind) -> f32 {
    let peak = ((1u64 << bits) - 1) as f32;
    let raw = match (full_range, kind) {
        (true, PlaneKind::Luma) => v * peak,
        (true, PlaneKind::Chroma) => (v + 0.5) * peak,
        (false, PlaneKind::Luma) => {
            let scale = (1u32 << (bits - 8)) as f32;
            v * 219.0 * scale + 16.0 * scale
        }
        (false, PlaneKind::Chroma) => {
            let scale = (1u32 << (bits - 8)) as f32;
            v * 224.0 * scale + 128.0 * scale
        }
    };
    raw.round().clamp(0.0, peak)
}

/// YUV → RGB in normalised `[0,1]`-ish units, non-constant-luminance.
///
/// `y`, `u`, `v` are expected already normalised by [`normalize_sample`].
#[inline]
pub fn yuv_to_rgb(y: f32, u: f32, v: f32, kr: f32, kb: f32) -> [f32; 3] {
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * v;
    let b = y + 2.0 * (1.0 - kb) * u;
    let g = y - (2.0 * (1.0 - kr) * kr / kg) * v - (2.0 * (1.0 - kb) * kb / kg) * u;
    [r, g, b]
}

/// RGB → YUV, inverse of [`yuv_to_rgb`].
#[inline]
pub fn rgb_to_yuv(r: f32, g: f32, b: f32, kr: f32, kb: f32) -> [f32; 3] {
    let kg = 1.0 - kr - kb;
    let y = kr * r + kg * g + kb * b;
    let u = (b - y) / (2.0 * (1.0 - kb));
    let v = (r - y) / (2.0 * (1.0 - kr));
    [y, u, v]
}

/// Linearises a BT.709 / BT.601 gamma-encoded value.
#[inline]
pub fn bt709_to_linear(e: f32) -> f32 {
    if e < 0.081 {
        e / 4.5
    } else {
        ((e + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// Inverse of [`bt709_to_linear`].
#[inline]
pub fn linear_to_bt709(l: f32) -> f32 {
    if l < 0.018 {
        l * 4.5
    } else {
        1.099 * l.powf(0.45) - 0.099
    }
}

/// Linearises an sRGB value.
#[inline]
pub fn srgb_to_linear(e: f32) -> f32 {
    if e <= 0.04045 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of [`srgb_to_linear`].
#[inline]
pub fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// Applies the requested transfer characteristic's EOTF (encode → linear).
#[inline]
pub fn to_linear(e: f32, t: u8) -> Result<f32, ColorError> {
    Ok(match t {
        transfer::LINEAR => e,
        transfer::BT709 => bt709_to_linear(e),
        transfer::SRGB => srgb_to_linear(e),
        other => return Err(ColorError::UnsupportedTransfer(other)),
    })
}

/// Applies the inverse of [`to_linear`] (linear → encode).
#[inline]
pub fn from_linear(l: f32, t: u8) -> Result<f32, ColorError> {
    Ok(match t {
        transfer::LINEAR => l,
        transfer::BT709 => linear_to_bt709(l),
        transfer::SRGB => linear_to_srgb(l),
        other => return Err(ColorError::UnsupportedTransfer(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_info_round_trip() {
        let s = SpaceInfo {
            matrix: matrix::BT709,
            range: range::LIMITED,
            transfer: transfer::BT709,
            primaries: primaries::BT709,
        };
        assert_eq!(SpaceInfo::from_bytes(&s.to_bytes()), s);
        assert!(!s.is_rgb());
        assert!(SpaceInfo::WORKING.is_rgb());
    }

    #[test]
    fn yuv_rgb_round_trip() {
        for m in [
            matrix::BT709,
            matrix::BT470BG,
            matrix::SMPTE170M,
            matrix::BT2020NC,
        ] {
            let (kr, kb) = luma_coefficients(m).unwrap();
            for &(r, g, b) in &[
                (0.0f32, 0.0f32, 0.0f32),
                (1.0, 1.0, 1.0),
                (0.8, 0.3, 0.55),
                (0.12, 0.9, 0.44),
            ] {
                let (y, u, v) = {
                    let t = rgb_to_yuv(r, g, b, kr, kb);
                    (t[0], t[1], t[2])
                };
                let back = yuv_to_rgb(y, u, v, kr, kb);
                assert!((back[0] - r).abs() < 1e-5, "m={m} r {r} -> {}", back[0]);
                assert!((back[1] - g).abs() < 1e-5, "m={m} g {g} -> {}", back[1]);
                assert!((back[2] - b).abs() < 1e-5, "m={m} b {b} -> {}", back[2]);
            }
        }
    }

    #[test]
    fn bt709_gamma_round_trip() {
        for i in 0..=100 {
            let e = i as f32 / 100.0;
            let back = linear_to_bt709(bt709_to_linear(e));
            assert!((back - e).abs() < 1e-5, "e={e} back={back}");
        }
    }

    #[test]
    fn srgb_gamma_round_trip() {
        for i in 0..=100 {
            let e = i as f32 / 100.0;
            let back = linear_to_srgb(srgb_to_linear(e));
            assert!((back - e).abs() < 1e-5, "e={e} back={back}");
        }
    }

    #[test]
    fn normalize_luma_full_and_limited() {
        assert!((normalize_sample(255.0, 8, true, PlaneKind::Luma) - 1.0).abs() < 1e-6);
        assert!((normalize_sample(0.0, 8, true, PlaneKind::Luma) - 0.0).abs() < 1e-6);
        assert!((normalize_sample(235.0, 8, false, PlaneKind::Luma) - 1.0).abs() < 1e-6);
        assert!((normalize_sample(16.0, 8, false, PlaneKind::Luma) - 0.0).abs() < 1e-6);
        assert!((normalize_sample(1023.0, 10, true, PlaneKind::Luma) - 1.0).abs() < 1e-6);
        assert!((normalize_sample(940.0, 10, false, PlaneKind::Luma) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn chroma_is_centred_on_zero() {
        // The property every YUV<->RGB conversion depends on. Getting this wrong offsets the whole
        // image by half the chroma range, which is exactly the bug this test exists to prevent.
        assert!((normalize_sample(128.0, 8, false, PlaneKind::Chroma) - 0.0).abs() < 1e-6);
        assert!((normalize_sample(16.0, 8, false, PlaneKind::Chroma) + 0.5).abs() < 1e-6);
        assert!((normalize_sample(240.0, 8, false, PlaneKind::Chroma) - 0.5).abs() < 1e-6);
        assert!((normalize_sample(512.0, 10, false, PlaneKind::Chroma) - 0.0).abs() < 1e-6);

        // Full range is centred too, but on 127.5 rather than 128: the span is [0, 255] mapped
        // linearly onto [-0.5, 0.5], so there is no integer neutral point at 8 bits.
        assert!((normalize_sample(127.5, 8, true, PlaneKind::Chroma) - 0.0).abs() < 1e-6);
        assert!((normalize_sample(0.0, 8, true, PlaneKind::Chroma) + 0.5).abs() < 1e-6);
        assert!((normalize_sample(255.0, 8, true, PlaneKind::Chroma) - 0.5).abs() < 1e-6);
        assert!((normalize_sample(128.0, 8, true, PlaneKind::Chroma) - 1.0 / 510.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_denormalize_round_trip() {
        for bits in [8u32, 10, 12, 16] {
            let peak = ((1u64 << bits) - 1) as f32;
            for &raw in &[0.0f32, peak * 0.25, peak * 0.5, peak] {
                for full in [true, false] {
                    for kind in [PlaneKind::Luma, PlaneKind::Chroma] {
                        let n = normalize_sample(raw, bits, full, kind);
                        let back = denormalize_sample(n, bits, full, kind);
                        assert!(
                            (back - raw).abs() <= 1.0,
                            "bits={bits} raw={raw} full={full} kind={kind:?} back={back}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn yuv_rgb_round_trip_through_sample_normalisation() {
        // The end-to-end property: an integer YUV triple must survive YUV -> RGB -> YUV.
        for bits in [8u32, 10u32] {
            for full in [true, false] {
                let (kr, kb) = luma_coefficients(matrix::BT709).unwrap();
                let peak = ((1u64 << bits) - 1) as f32;
                let samples = [
                    (0.40 * peak, 0.50 * peak, 0.50 * peak),
                    (0.20 * peak, 0.25 * peak, 0.75 * peak),
                    (0.80 * peak, 0.70 * peak, 0.30 * peak),
                ];
                for &(y_raw, u_raw, v_raw) in &samples {
                    let y = normalize_sample(y_raw, bits, full, PlaneKind::Luma);
                    let u = normalize_sample(u_raw, bits, full, PlaneKind::Chroma);
                    let v = normalize_sample(v_raw, bits, full, PlaneKind::Chroma);
                    let rgb = yuv_to_rgb(y, u, v, kr, kb);
                    let back = rgb_to_yuv(rgb[0], rgb[1], rgb[2], kr, kb);
                    let y2 = denormalize_sample(back[0], bits, full, PlaneKind::Luma);
                    let u2 = denormalize_sample(back[1], bits, full, PlaneKind::Chroma);
                    let v2 = denormalize_sample(back[2], bits, full, PlaneKind::Chroma);
                    assert!(
                        (y2 - y_raw).abs() <= 1.0,
                        "bits={bits} full={full} Y {y_raw} -> {y2}"
                    );
                    assert!(
                        (u2 - u_raw).abs() <= 1.0,
                        "bits={bits} full={full} U {u_raw} -> {u2}"
                    );
                    assert!(
                        (v2 - v_raw).abs() <= 1.0,
                        "bits={bits} full={full} V {v_raw} -> {v2}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_unsupported_codes() {
        assert!(luma_coefficients(matrix::BT2020C).is_err());
        assert!(luma_coefficients(matrix::UNSPECIFIED).is_err());
        assert!(to_linear(0.5, transfer::UNSPECIFIED).is_err());
        assert!(to_linear(0.5, transfer::ST2084).is_err());
    }
}

/// The affine map between one plane's native samples and the normalised working space.
///
/// `normalise_sample` and `denormalize_sample` are branchy because they take the range and the plane
/// kind separately; the GPU needs the same thing as constants it can multiply by. Both forms are the
/// same arithmetic — this is the affine expansion of those two functions, and `plane_scale_matches`
/// below holds them to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneScale {
    /// Multiplier in `normalised = clamp(raw * norm_scale + norm_offset, norm_lo, norm_hi)`.
    pub norm_scale: f32,
    /// Addend in the same expression.
    pub norm_offset: f32,
    /// Lower clamp bound on the normalised value.
    pub norm_lo: f32,
    /// Upper clamp bound on the normalised value.
    pub norm_hi: f32,
    /// Multiplier in `raw = round(clamp(normalised * denorm_scale + denorm_offset, 0, peak))`.
    pub denorm_scale: f32,
    /// Addend in the same expression.
    pub denorm_offset: f32,
    /// Largest representable native sample, which is also the denormalised clamp bound.
    pub peak: f32,
}

/// The affine constants for one `(bits, range, plane kind)` combination.
pub fn plane_scale(bits: u32, full_range: bool, kind: PlaneKind) -> PlaneScale {
    let peak = ((1u64 << bits) - 1) as f32;
    // A clamp that cannot fire, for the cases `normalize_sample` does not clamp.
    const WIDE: f32 = 3.4e38;
    match (full_range, kind) {
        (true, PlaneKind::Luma) => PlaneScale {
            norm_scale: 1.0 / peak,
            norm_offset: 0.0,
            norm_lo: -WIDE,
            norm_hi: WIDE,
            denorm_scale: peak,
            denorm_offset: 0.0,
            peak,
        },
        (true, PlaneKind::Chroma) => PlaneScale {
            norm_scale: 1.0 / peak,
            norm_offset: -0.5,
            norm_lo: -1.0,
            norm_hi: 1.0,
            denorm_scale: peak,
            denorm_offset: 0.5 * peak,
            peak,
        },
        (false, PlaneKind::Luma) => {
            let s = (1u32 << (bits - 8)) as f32;
            PlaneScale {
                norm_scale: 1.0 / (219.0 * s),
                norm_offset: -16.0 / 219.0,
                norm_lo: -0.5,
                norm_hi: 1.5,
                denorm_scale: 219.0 * s,
                denorm_offset: 16.0 * s,
                peak,
            }
        }
        (false, PlaneKind::Chroma) => {
            let s = (1u32 << (bits - 8)) as f32;
            PlaneScale {
                norm_scale: 1.0 / (224.0 * s),
                norm_offset: -128.0 / 224.0,
                norm_lo: -1.0,
                norm_hi: 1.0,
                denorm_scale: 224.0 * s,
                denorm_offset: 128.0 * s,
                peak,
            }
        }
    }
}

#[cfg(test)]
mod plane_scale_tests {
    use super::*;

    /// The affine form must reproduce the branchy reference exactly, or the GPU path would disagree
    /// with the CPU on every sample.
    #[test]
    fn plane_scale_matches_normalize_and_denormalize() {
        for bits in [8u32, 10, 12, 16] {
            for full_range in [false, true] {
                for kind in [PlaneKind::Luma, PlaneKind::Chroma] {
                    let s = plane_scale(bits, full_range, kind);
                    let peak = ((1u64 << bits) - 1) as f32;
                    let mut raw = 0.0f32;
                    while raw <= peak {
                        let expected = normalize_sample(raw, bits, full_range, kind);
                        let got = (raw * s.norm_scale + s.norm_offset).clamp(s.norm_lo, s.norm_hi);
                        assert!(
                            (expected - got).abs() <= 1e-6,
                            "bits={bits} full={full_range} {kind:?} raw={raw}: {expected} vs {got}"
                        );
                        let back = denormalize_sample(expected, bits, full_range, kind);
                        let affine = (got * s.denorm_scale + s.denorm_offset)
                            .clamp(0.0, peak)
                            .round();
                        assert!(
                            (back - affine).abs() <= 1.0,
                            "bits={bits} full={full_range} {kind:?} raw={raw}: back {back} vs {affine}"
                        );
                        raw += (peak / 97.0).max(1.0);
                    }
                }
            }
        }
    }
}

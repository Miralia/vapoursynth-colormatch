//! Model representation and `.cm` serialization.
//!
//! The on-disk / on-the-wire format is little-endian and self-describing:
//!
//! ```text
//! offset  size  field
//!      0     4  magic "CMAT"
//!      4     2  format version
//!      6     2  flags (reserved, must be 0)
//!      8     4  nodes            (uniform grid nodes per axis, 257)
//!     12     4  n_slices         (K)
//!     16     8  SpaceInfo test   (4 × u8) + SpaceInfo ref (4 × u8)
//!     24     4  transfer encoding hint (reserved)
//!     28     4  directions
//!     32     8  samples          (u64)
//!     40     4  frames
//!     44     4  w1_before        (f32)
//!     48     4  w1_after         (f32)
//!     52     4  max_gain         (f32)
//!     56     4  build_ms         (f32)
//!     60     4  reserved (must be 0)
//!     64    ..  slices
//! ```
//!
//! Each slice is `9 × f32` (row-major rotation) followed by three axes, each
//! `lo: f32, scale: f32, tab: [f32; nodes]`.

use crate::color::SpaceInfo;

/// File magic: `CMAT`.
pub const MAGIC: [u8; 4] = *b"CMAT";

/// Current `.cm` format version.
pub const FORMAT_VERSION: u16 = 1;

/// Number of uniform grid nodes per axis. `nodes - 1` is the bin count `B`.
pub const TABLE_NODES: usize = 257;

/// Size of the fixed header in bytes.
pub const HEADER_LEN: usize = 64;

/// Errors produced while parsing a serialized model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// The buffer is shorter than the fixed header.
    TruncatedHeader,
    /// The magic bytes are not `CMAT`.
    BadMagic([u8; 4]),
    /// The format version is not supported by this build.
    UnsupportedVersion(u16),
    /// Reserved fields were non-zero.
    ReservedNotZero,
    /// The node count is not supported.
    UnsupportedNodes(u32),
    /// The buffer ended in the middle of a slice.
    TruncatedSlices,
    /// The slice count would overflow.
    TooManySlices(u32),
    /// A rotation matrix was not orthonormal (det or norm out of tolerance).
    NonOrthonormalSlice(usize),
    /// An axis table was not monotone non-decreasing.
    NonMonotoneAxis {
        /// Index of the offending slice.
        slice: usize,
        /// Index of the offending axis within the slice.
        axis: usize,
    },
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TruncatedHeader => {
                write!(f, "buffer is shorter than the {HEADER_LEN}-byte header")
            }
            Self::BadMagic(m) => write!(f, "bad magic {m:?}, expected {MAGIC:?}"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported .cm format version {v} (this build supports {FORMAT_VERSION})"
                )
            }
            Self::ReservedNotZero => write!(f, "reserved header fields are non-zero"),
            Self::UnsupportedNodes(n) => {
                write!(f, "unsupported node count {n}, expected {TABLE_NODES}")
            }
            Self::TruncatedSlices => write!(f, "buffer ended in the middle of a slice"),
            Self::TooManySlices(n) => write!(f, "implausible slice count {n}"),
            Self::NonOrthonormalSlice(i) => {
                write!(f, "slice {i} rotation matrix is not orthonormal")
            }
            Self::NonMonotoneAxis { slice, axis } => {
                write!(
                    f,
                    "slice {slice} axis {axis} transfer table is not monotone"
                )
            }
        }
    }
}

impl std::error::Error for ModelError {}

/// One projected axis: a monotone transfer function on a uniform grid.
///
/// Evaluation is `i = clamp(((v - lo) * scale) as usize, 0, nodes - 2)` followed by a linear
/// interpolation between `tab[i]` and `tab[i + 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    /// Grid origin.
    pub lo: f32,
    /// Grid scale: `scale = (nodes - 1) / (hi - lo)`.
    pub scale: f32,
    /// Transfer values, one per grid node. Must be monotone non-decreasing.
    ///
    /// Stored inline rather than behind a `Vec` on purpose: the apply inner loop reads this once per
    /// axis per slice per pixel, so a pointer chase and a bounds check against a runtime length are
    /// measurable. With a fixed-size array the length is a compile-time constant and the table sits
    /// directly in the slice struct.
    pub tab: [f32; TABLE_NODES],
}

impl Axis {
    /// Builds an axis from an explicit grid and a value-producing closure.
    ///
    /// The closure is evaluated at [`TABLE_NODES`] evenly spaced points in `[lo, hi]`.
    pub fn from_fn(lo: f32, hi: f32, mut f: impl FnMut(f32) -> f32) -> Self {
        let scale = (TABLE_NODES - 1) as f32 / (hi - lo);
        let mut tab = [0.0f32; TABLE_NODES];
        for (i, slot) in tab.iter_mut().enumerate() {
            *slot = f(lo + i as f32 / scale);
        }
        Self { lo, scale, tab }
    }

    /// Builds an identity axis over `[lo, hi]`.
    pub fn identity(lo: f32, hi: f32) -> Self {
        Self::from_fn(lo, hi, |x| x)
    }

    /// Number of grid nodes.
    pub const fn nodes(&self) -> usize {
        TABLE_NODES
    }

    /// Grid upper bound (inclusive).
    pub fn hi(&self) -> f32 {
        self.lo + (TABLE_NODES - 1) as f32 / self.scale
    }

    /// Evaluates the transfer function with a clamped uniform-grid lookup.
    ///
    /// Values outside the grid return the end nodes exactly (rather than the second-to-last node),
    /// which matters because the builder appends `±√3` sentinels that must be reproduced.
    /// Non-finite inputs collapse to the low node instead of propagating NaN.
    #[inline]
    /// Evaluates the transfer function, **extrapolating with unit slope outside the grid**.
    ///
    /// The grid is finite — `[0, 1]` for the first slice, `[-sqrt(3), sqrt(3)]` afterwards — but the
    /// values it is applied to need not be. Limited-range YUV converts to RGB outside `[0, 1]` by
    /// design: `Y=235` with non-neutral chroma is above 1.0, and the standard reserves the range for
    /// undershoot and overshoot.
    ///
    /// Clamping those values to the grid's end node is not a harmless edge case. Measured on a
    /// limited-range 4:2:0 self-match, where the correct answer is the identity and the fitted tables
    /// *were* exactly the identity, clamping values outside `[0, 1]` still left a residual of 1.6e-5
    /// on the first slice alone. Because the sweep composes slices, that residual then compounded:
    /// the worst-case gain reached 36 across 32 slices, and an identity model shifted 4:2:0 samples by
    /// 5% of full scale.
    ///
    /// Extrapolating with slope 1 makes the function the identity wherever the grid has no opinion,
    /// which is the same principle as the identity tail ramps inside the grid: content the model was
    /// not fitted on passes through untouched. Monotonicity is preserved because the slope outside is
    /// exactly 1, and the table's end nodes are reached continuously from within.
    pub fn eval(&self, v: f32) -> f32 {
        const LAST: usize = TABLE_NODES - 1;
        let idx = (v - self.lo) * self.scale;
        if !idx.is_finite() {
            // NaN, +inf and -inf all have to land somewhere finite and deterministic. Continuing
            // with unit slope is not possible for a non-finite input, so return the input's own
            // sign-appropriate extreme of the table.
            return if idx > 0.0 {
                self.tab[LAST]
            } else {
                self.tab[0]
            };
        }
        if idx <= 0.0 {
            // Unit slope: `v` is one grid step per unit of `idx` below the origin.
            return self.tab[0] + (idx * (1.0 / self.scale));
        }
        if idx >= LAST as f32 {
            return self.tab[LAST] + ((idx - LAST as f32) * (1.0 / self.scale));
        }
        let i = idx as usize;
        let f = idx - i as f32;
        let a = self.tab[i];
        a + f * (self.tab[i + 1] - a)
    }

    /// Returns `true` if the table is monotone non-decreasing (within `tol`).
    pub fn is_monotone(&self, tol: f32) -> bool {
        self.tab.windows(2).all(|w| w[1] >= w[0] - tol)
    }
}

/// One slice of the transform: a rotation plus three per-axis transfer functions.
#[derive(Debug, Clone, PartialEq)]
pub struct Slice {
    /// Row-major rotation matrix. Must be orthonormal so that `Rᵀ == R⁻¹`.
    pub rot: [[f32; 3]; 3],
    /// The three per-axis transfer functions, indexed by rotated channel.
    pub axes: [Axis; 3],
}

impl Slice {
    /// The identity slice (identity rotation, identity axes over `[lo, hi]`).
    pub fn identity(lo: f32, hi: f32) -> Self {
        Self {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [
                Axis::identity(lo, hi),
                Axis::identity(lo, hi),
                Axis::identity(lo, hi),
            ],
        }
    }

    /// Determinant of the rotation matrix; `+1` for a proper rotation.
    pub fn determinant(&self) -> f32 {
        let m = &self.rot;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Maximum deviation of `RᵀR` from the identity.
    pub fn orthonormality_error(&self) -> f32 {
        let mut worst = 0.0f32;
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += self.rot[k][i] * self.rot[k][j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((acc - expect).abs());
            }
        }
        worst
    }
}

/// Statistics recorded while building the model, surfaced as frame properties.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelStats {
    /// Number of pooled sample pixels used for the build.
    pub samples: u64,
    /// Number of source frames sampled across all inputs.
    pub frames: u32,
    /// Mean sliced-Wasserstein-1 distance between test and reference before the transform.
    pub w1_before: f32,
    /// Same distance after the transform (measured on the build samples).
    pub w1_after: f32,
    /// Largest estimated local gain (max singular value of the Jacobian) on the sample cloud.
    pub max_gain: f32,
    /// Wall-clock build time in milliseconds.
    pub build_ms: f32,
    /// Number of directions actually used.
    pub directions: u32,
}

impl ModelStats {
    /// Packs the stats into the flat `f64` array stored in the `_CM_Stats` frame property.
    ///
    /// Layout: `[samples, frames, w1_before, w1_after, max_gain, build_ms, directions]`.
    pub fn as_props(&self) -> [f64; 7] {
        [
            self.samples as f64,
            self.frames as f64,
            self.w1_before as f64,
            self.w1_after as f64,
            self.max_gain as f64,
            self.build_ms as f64,
            self.directions as f64,
        ]
    }

    /// Names of the fields in [`ModelStats::as_props`], in order.
    pub const PROP_NAMES: [&'static str; 7] = [
        "samples",
        "frames",
        "w1_before",
        "w1_after",
        "max_gain",
        "build_ms",
        "directions",
    ];
}

/// A complete colour transform: a list of slices plus the colour spaces it was built for.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// Format version of the container this model was read from / will be written as.
    pub version: u16,
    /// Grid node count shared by every axis in every slice.
    pub nodes: u32,
    /// Colour space of the test (source) side.
    pub test_space: SpaceInfo,
    /// Colour space of the reference (target) side.
    pub ref_space: SpaceInfo,
    /// Build statistics.
    pub stats: ModelStats,
    /// The ordered slices. Applying the model runs them in order.
    pub slices: Vec<Slice>,
}

impl Model {
    /// An empty model: applying it is the identity.
    pub fn identity(nodes: usize) -> Self {
        Self {
            version: FORMAT_VERSION,
            nodes: nodes as u32,
            test_space: SpaceInfo::default(),
            ref_space: SpaceInfo::default(),
            stats: ModelStats::default(),
            slices: Vec::new(),
        }
    }

    /// Serializes the model into the `.cm` wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let nodes = self.nodes as usize;
        let mut out =
            Vec::with_capacity(HEADER_LEN + self.slices.len() * (36 + 3 * (8 + TABLE_NODES * 4)));

        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&self.nodes.to_le_bytes());
        out.extend_from_slice(&(self.slices.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.test_space.to_bytes());
        out.extend_from_slice(&self.ref_space.to_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // transfer hint (reserved)
        out.extend_from_slice(&self.stats.directions.to_le_bytes());
        out.extend_from_slice(&self.stats.samples.to_le_bytes());
        out.extend_from_slice(&self.stats.frames.to_le_bytes());
        out.extend_from_slice(&self.stats.w1_before.to_le_bytes());
        out.extend_from_slice(&self.stats.w1_after.to_le_bytes());
        out.extend_from_slice(&self.stats.max_gain.to_le_bytes());
        out.extend_from_slice(&self.stats.build_ms.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
        debug_assert_eq!(out.len(), HEADER_LEN);

        for s in &self.slices {
            for row in &s.rot {
                for v in row {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            for a in &s.axes {
                out.extend_from_slice(&a.lo.to_le_bytes());
                out.extend_from_slice(&a.scale.to_le_bytes());
                debug_assert_eq!(a.tab.len(), nodes);
                for v in a.tab.iter() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Parses the `.cm` wire format.
    ///
    /// Validation is deliberately strict: a malformed model is rejected rather than silently
    /// producing a wrong grade.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, ModelError> {
        if buf.len() < HEADER_LEN {
            return Err(ModelError::TruncatedHeader);
        }
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(ModelError::BadMagic(magic));
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }
        let flags = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let nodes = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let n_slices = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        if flags != 0 {
            return Err(ModelError::ReservedNotZero);
        }
        if nodes as usize != TABLE_NODES {
            return Err(ModelError::UnsupportedNodes(nodes));
        }
        if n_slices > 4096 {
            return Err(ModelError::TooManySlices(n_slices));
        }
        let test_space = SpaceInfo::from_bytes(&buf[16..20]);
        let ref_space = SpaceInfo::from_bytes(&buf[20..24]);
        if u32::from_le_bytes(buf[24..28].try_into().unwrap()) != 0 {
            return Err(ModelError::ReservedNotZero);
        }
        let directions = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let samples = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let frames = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let w1_before = f32::from_le_bytes(buf[44..48].try_into().unwrap());
        let w1_after = f32::from_le_bytes(buf[48..52].try_into().unwrap());
        let max_gain = f32::from_le_bytes(buf[52..56].try_into().unwrap());
        let build_ms = f32::from_le_bytes(buf[56..60].try_into().unwrap());
        if u32::from_le_bytes(buf[60..64].try_into().unwrap()) != 0 {
            return Err(ModelError::ReservedNotZero);
        }

        let nodes = nodes as usize;
        let slice_len = 9 * 4 + 3 * (2 * 4 + nodes * 4);
        let need = n_slices as usize * slice_len;
        if buf.len() < HEADER_LEN + need {
            return Err(ModelError::TruncatedSlices);
        }

        let mut slices = Vec::with_capacity(n_slices as usize);
        let mut off = HEADER_LEN;
        let f32_at = |off: &mut usize| -> f32 {
            let v = f32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };
        for si in 0..n_slices as usize {
            let mut rot = [[0f32; 3]; 3];
            for row in rot.iter_mut() {
                for v in row.iter_mut() {
                    *v = f32_at(&mut off);
                }
            }
            let mut axes = Vec::with_capacity(3);
            for _ in 0..3 {
                let lo = f32_at(&mut off);
                let scale = f32_at(&mut off);
                let mut tab = [0.0f32; TABLE_NODES];
                for slot in tab.iter_mut() {
                    *slot = f32_at(&mut off);
                }
                axes.push(Axis { lo, scale, tab });
            }
            let axes: [Axis; 3] = axes.try_into().map_err(|_| ModelError::TruncatedSlices)?;
            let slice = Slice { rot, axes };
            if slice.orthonormality_error() > 1e-3 || (slice.determinant() - 1.0).abs() > 1e-3 {
                return Err(ModelError::NonOrthonormalSlice(si));
            }
            for (ai, a) in slice.axes.iter().enumerate() {
                if !a.is_monotone(1e-4) {
                    return Err(ModelError::NonMonotoneAxis {
                        slice: si,
                        axis: ai,
                    });
                }
            }
            slices.push(slice);
        }

        Ok(Self {
            version,
            nodes: nodes as u32,
            test_space,
            ref_space,
            stats: ModelStats {
                samples,
                frames,
                w1_before,
                w1_after,
                max_gain,
                build_ms,
                directions,
            },
            slices,
        })
    }

    /// Number of slices (`K`).
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// Returns `true` when the model is the identity.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Serialized size in bytes.
    pub fn serialized_len(&self) -> usize {
        HEADER_LEN + self.slices.len() * (9 * 4 + 3 * (2 * 4 + TABLE_NODES * 4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model(k: usize) -> Model {
        let mut m = Model::identity(TABLE_NODES);
        m.stats = ModelStats {
            samples: 262_144,
            frames: 64,
            w1_before: 0.0385,
            w1_after: 0.00046,
            max_gain: 2.33,
            build_ms: 118.5,
            directions: k as u32,
        };
        m.test_space = SpaceInfo {
            matrix: 1,
            range: 1,
            transfer: 1,
            primaries: 1,
        };
        m.ref_space = SpaceInfo {
            matrix: 0,
            range: 1,
            transfer: 1,
            primaries: 1,
        };
        for i in 0..k {
            let mut s = Slice::identity(-1.732_050_8, 1.732_050_8);
            // A plausible small rotation so the orthonormality check is exercised.
            let t = (i as f32 + 1.0) * 0.037;
            let (c, sn) = (t.cos(), t.sin());
            s.rot = [[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]];
            // A monotone non-identity transfer.
            for (ai, a) in s.axes.iter_mut().enumerate() {
                for (j, v) in a.tab.iter_mut().enumerate() {
                    let x = a.lo + j as f32 / a.scale;
                    *v = x * (1.0 + 0.01 * ai as f32) + 0.001 * (i as f32);
                }
            }
            m.slices.push(s);
        }
        m
    }

    #[test]
    fn axis_eval_interpolates_and_is_monotone() {
        let a = Axis::identity(0.0, 1.0);
        assert_eq!(a.nodes(), TABLE_NODES);
        assert!((a.eval(0.5) - 0.5).abs() < 1e-3);
        // Midpoint of a node interval should interpolate, not snap.
        let step = 1.0 / 256.0;
        let v = a.eval(step * 0.5);
        assert!((v - step * 0.5).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn axis_eval_extrapolates_with_unit_slope_outside_the_grid() {
        // Limited-range YUV converts to RGB outside [0, 1] by design, so this is the normal case
        // rather than an edge case. Clamping here left a per-slice residual that compounded into a
        // 36x worst-case gain and a 5%-of-full-scale shift on an identity 4:2:0 model.
        let a = Axis::identity(0.0, 1.0);
        assert!((a.eval(-0.25) + 0.25).abs() < 1e-5, "{}", a.eval(-0.25));
        assert!((a.eval(1.5) - 1.5).abs() < 1e-5, "{}", a.eval(1.5));

        // Continuity at both ends: no jump between the last in-grid value and the first outside it.
        assert!((a.eval(1.0 + 1e-5) - a.eval(1.0 - 1e-5)).abs() < 1e-4);
        assert!((a.eval(-1e-5) - a.eval(1e-5)).abs() < 1e-4);

        // Monotone throughout, which is what keeps the composed transform well behaved.
        let mut prev = f32::MIN;
        let mut v = -2.0f32;
        while v <= 2.0 {
            let y = a.eval(v);
            assert!(y >= prev - 1e-6, "not monotone at {v}: {prev} -> {y}");
            prev = y;
            v += 0.001;
        }
    }

    #[test]
    fn self_match_stays_at_the_identity_outside_the_grid() {
        // The end-to-end property behind the fix: fitting a cloud against itself must produce the
        // identity even when the cloud extends past the slice-0 grid, which limited-range YUV does.
        use crate::build::{BuildOptions, Samples, build_model};

        let mut cloud = Vec::new();
        let mut s = 0x9E37_79B9u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as f32 / u32::MAX as f32
        };
        for _ in 0..20_000 {
            // A cloud that reaches below 0 and above 1 on every channel.
            cloud.push([
                -0.15 + 1.2 * rnd(),
                -0.15 + 1.2 * rnd(),
                -0.15 + 1.2 * rnd(),
            ]);
        }
        for k in [1usize, 4, 16, 32] {
            let m = build_model(
                &Samples::new(cloud.clone(), 1),
                &Samples::new(cloud.clone(), 1),
                &BuildOptions {
                    directions: k,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                m.stats.w1_after < 1e-5,
                "K={k}: w1_after={}",
                m.stats.w1_after
            );
            assert!(
                m.stats.max_gain < 1.5,
                "K={k}: a self-match must stay near the identity, got gain {}",
                m.stats.max_gain
            );
        }
    }

    #[test]
    fn axis_eval_handles_nan() {
        let a = Axis::identity(0.0, 1.0);
        assert!(a.eval(f32::NAN).is_finite());
        assert!(a.eval(f32::INFINITY).is_finite());
        assert!(a.eval(f32::NEG_INFINITY).is_finite());
    }

    #[test]
    fn round_trip_is_lossless() {
        let m = sample_model(32);
        let bytes = m.to_bytes();
        assert_eq!(bytes.len(), m.serialized_len());
        let back = Model::from_bytes(&bytes).expect("parse");
        assert_eq!(back, m);
    }

    #[test]
    fn identity_model_is_about_100kb_at_k32() {
        let m = sample_model(32);
        let n = m.serialized_len();
        assert!((99_000..103_000).contains(&n), "unexpected size {n}");
    }

    #[test]
    fn rejects_corruption() {
        let m = sample_model(2);
        let good = m.to_bytes();

        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(matches!(
            Model::from_bytes(&bad),
            Err(ModelError::BadMagic(_))
        ));

        let mut bad = good.clone();
        bad[4] = 99;
        assert!(matches!(
            Model::from_bytes(&bad),
            Err(ModelError::UnsupportedVersion(_))
        ));

        let mut bad = good.clone();
        bad[6] = 1;
        assert!(matches!(
            Model::from_bytes(&bad),
            Err(ModelError::ReservedNotZero)
        ));

        assert!(matches!(
            Model::from_bytes(&good[..10]),
            Err(ModelError::TruncatedHeader)
        ));
        assert!(matches!(
            Model::from_bytes(&good[..good.len() - 4]),
            Err(ModelError::TruncatedSlices)
        ));

        // Break monotonicity of the first axis table.
        let mut bad = good.clone();
        let first_tab = HEADER_LEN + 36 + 8;
        let v0 = f32::from_le_bytes(bad[first_tab..first_tab + 4].try_into().unwrap());
        bad[first_tab + 4..first_tab + 8].copy_from_slice(&(v0 - 1.0).to_le_bytes());
        assert!(matches!(
            Model::from_bytes(&bad),
            Err(ModelError::NonMonotoneAxis { .. })
        ));

        // Break orthonormality of the first slice.
        let mut bad = good.clone();
        bad[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&2.0f32.to_le_bytes());
        assert!(matches!(
            Model::from_bytes(&bad),
            Err(ModelError::NonOrthonormalSlice(0))
        ));
    }

    #[test]
    fn slice_orthonormality_check() {
        let mut s = Slice::identity(0.0, 1.0);
        assert!(s.orthonormality_error() < 1e-6);
        assert!((s.determinant() - 1.0).abs() < 1e-6);
        s.rot[0][0] = 1.5;
        assert!(s.orthonormality_error() > 1e-3);
    }

    #[test]
    fn empty_model_round_trips() {
        let m = Model::identity(TABLE_NODES);
        assert!(m.is_empty());
        let back = Model::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.serialized_len(), HEADER_LEN);
    }
}

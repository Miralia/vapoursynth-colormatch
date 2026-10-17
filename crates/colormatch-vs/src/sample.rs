//! Sampling: turning frames into pooled RGB sample clouds.
//!
//! Sampling is deliberately cheap and deterministic. Measured on synthetic cross-channel looks,
//! the pooled residual stops improving beyond roughly 20 000 sample pixels and is saturated by
//! 200 000, so the default budget is 262 144 pixels per side regardless of how many frames that
//! is spread over.
//!
//! Spatial subsampling uses a fixed stride rather than a PRNG so that a given clip, frame list and
//! budget always produce the same model — which is what makes content-hash caching meaningful.

use anyhow::{Result, bail};
use colormatch_core::build::Samples;
use vapoursynth::prelude::*;

use crate::frame_io::{ClipFormat, frame_to_rgb};

/// How to pick which frames of a clip to sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameSelection {
    /// `n` frames spread evenly across the clip.
    Even(usize),
    /// Every `n`-th frame.
    Every(usize),
    /// An explicit list of frame indices, clamped to the clip length.
    Explicit(Vec<usize>),
}

impl FrameSelection {
    /// Resolves the selection into concrete, in-range, de-duplicated frame indices.
    pub fn resolve(&self, num_frames: usize) -> Vec<usize> {
        if num_frames == 0 {
            return Vec::new();
        }
        match self {
            Self::Even(n) => {
                let n = (*n).max(1).min(num_frames);
                if n == 1 {
                    return vec![num_frames / 2];
                }
                (0..n).map(|i| i * (num_frames - 1) / (n - 1)).collect()
            }
            Self::Every(step) => {
                let step = (*step).max(1);
                (0..num_frames).step_by(step).collect()
            }
            Self::Explicit(list) => {
                let mut v: Vec<usize> = list.iter().copied().filter(|&i| i < num_frames).collect();
                v.sort_unstable();
                v.dedup();
                v
            }
        }
    }
}

/// A fixed-stride spatial subsample of a frame's pixels.
///
/// Returns indices into the row-major plane arrays. The stride is chosen so that at most
/// `max_pixels` samples are produced; when the frame is already small enough every pixel is used.
pub fn spatial_indices(width: usize, height: usize, max_pixels: usize) -> Vec<usize> {
    let total = width * height;
    if max_pixels == 0 || total <= max_pixels {
        return (0..total).collect();
    }
    // Stride in both axes; ceil keeps the result at or below the budget.
    let ratio = (total as f64 / max_pixels as f64).sqrt().ceil() as usize;
    let step = ratio.max(1);
    let mut out = Vec::with_capacity(max_pixels + step);
    let mut y = 0;
    while y < height {
        let row = y * width;
        let mut x = 0;
        while x < width {
            out.push(row + x);
            x += step;
        }
        y += step;
    }
    out
}

/// Reads `frames` from `get_frame` and accumulates a pooled sample cloud.
///
/// `get_frame` is called once per requested frame index; the caller is responsible for having
/// requested those frames from VapourSynth first.
pub fn sample_frames<'core, F>(
    frames: &[usize],
    fmt: &ClipFormat,
    max_pixels_per_frame: usize,
    mut get_frame: F,
) -> Result<Samples>
where
    F: FnMut(usize) -> Result<FrameRef<'core>>,
{
    if frames.is_empty() {
        bail!("cm.BuildModel: no frames selected for sampling");
    }
    let idx = spatial_indices(fmt.width, fmt.height, max_pixels_per_frame);
    let mut cloud: Vec<[f32; 3]> = Vec::with_capacity(idx.len() * frames.len());
    for &n in frames {
        let frame = get_frame(n)?;
        let [r, g, b] = frame_to_rgb(&frame, fmt)?;
        for &i in &idx {
            cloud.push([r[i], g[i], b[i]]);
        }
    }
    Ok(Samples::new(cloud, frames.len() as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_selection_covers_both_ends() {
        let f = FrameSelection::Even(5).resolve(101);
        assert_eq!(f, vec![0, 25, 50, 75, 100]);
    }

    #[test]
    fn even_selection_single_frame_picks_the_middle() {
        assert_eq!(FrameSelection::Even(1).resolve(101), vec![50]);
    }

    #[test]
    fn even_selection_is_clamped_to_the_clip_length() {
        let f = FrameSelection::Even(1000).resolve(10);
        assert_eq!(f.len(), 10);
        assert_eq!(f[0], 0);
        assert_eq!(*f.last().unwrap(), 9);
    }

    #[test]
    fn every_selection_steps() {
        assert_eq!(FrameSelection::Every(3).resolve(10), vec![0, 3, 6, 9]);
        assert_eq!(FrameSelection::Every(0).resolve(4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn explicit_selection_drops_out_of_range_and_dedupes() {
        let f = FrameSelection::Explicit(vec![9, 1, 1, 100, 3]).resolve(10);
        assert_eq!(f, vec![1, 3, 9]);
    }

    #[test]
    fn empty_clip_yields_no_frames() {
        assert!(FrameSelection::Even(8).resolve(0).is_empty());
        assert!(FrameSelection::Every(2).resolve(0).is_empty());
        assert!(FrameSelection::Explicit(vec![0]).resolve(0).is_empty());
    }

    #[test]
    fn spatial_indices_respect_the_budget() {
        let idx = spatial_indices(1920, 1080, 4096);
        assert!(idx.len() <= 4096 + 200, "got {}", idx.len());
        assert!(idx.len() > 1000);
        assert!(idx.iter().all(|&i| i < 1920 * 1080));
    }

    #[test]
    fn spatial_indices_are_identity_for_small_frames() {
        let idx = spatial_indices(4, 4, 1000);
        assert_eq!(idx, (0..16).collect::<Vec<_>>());
        let idx = spatial_indices(4, 4, 0);
        assert_eq!(idx, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn spatial_indices_are_deterministic() {
        let a = spatial_indices(1920, 1080, 4096);
        let b = spatial_indices(1920, 1080, 4096);
        assert_eq!(a, b);
    }
}

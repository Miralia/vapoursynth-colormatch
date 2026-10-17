# vapoursynth-colormatch task runner.
#
# `just` is not required — every recipe is a short shell command you can run by hand. It exists so
# that the common flows are one word, matching how vs-jetpack is driven.

set shell := ["bash", "-cu"]

plugin_darwin  := "target/release/libcolormatch.dylib"
plugin_linux   := "target/release/libcolormatch.so"
plugin_windows := "target/release/colormatch.dll"

# Build the plugin and the CLI in release mode.
build:
    cargo build --release

# Build only the VapourSynth plugin.
plugin:
    cargo build --release -p colormatch-vs

# Rust unit and integration tests. No VapourSynth required.
test:
    cargo test

# Rust tests plus clippy with warnings denied.
lint:
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check

# The VapourSynth integration suite: builds the plugin first, then drives it through `vapoursynth`.
test-integration: plugin
    python3 tests/integration/test_plugin_e2e.py

# Python wrapper tests. Needs the plugin built and $VSCOLORMATCH_PLUGIN pointing at it for the
# plugin-backed tests; the rest run without.
test-python:
    PYTHONPATH=python/src python3 -m pytest -q tests/python

# Everything.
test-all: test test-integration test-python

# Inspect a serialized model.
inspect path:
    cargo run --release -p colormatch-cli -- inspect {{path}}

# Export a model as a 3D LUT.
cube model out size="33":
    cargo run --release -p colormatch-cli -- cube {{model}} {{out}} --size {{size}}

# Build a wheel for the host platform, with the plugin staged inside it.
wheel:
    uv build

# Build the wheel, install it into a throwaway venv, and check that the plugin loads and runs.
#
# Worth running after any packaging change: the wheel can be structurally wrong in ways that `uv
# build` alone will not tell you. It has been wrong in three: an `any` platform tag on a wheel
# containing a dylib, a duplicate copy of the plugin from a bad `parent` count, and a version source
# that could not resolve at all without a git tag.
test-wheel: wheel
    rm -rf /tmp/vscm-wheel-check dist-wheel
    python3 -m venv --system-site-packages /tmp/vscm-wheel-check
    /tmp/vscm-wheel-check/bin/pip install -q --force-reinstall dist/*.whl
    /tmp/vscm-wheel-check/bin/python -c "\
import vapoursynth as vs;\
import vs_colormatch as vscm;\
vscm.load_plugin();\
assert hasattr(vs.core, 'cm'), 'the cm namespace did not load';\
c = vs.core.std.BlankClip(width=64, height=64, format=vs.RGB48, length=4);\
m = vs.core.cm.BuildModel(test=[c], ref=[c], directions=8);\
assert m.get_frame(0).props['_CM_ModelBytes'] > 0;\
out = vscm.match(c, c, directions=8);\
print('wheel ok:', vscm.__version__, out.format.name, out.num_frames, 'frames');\
"
    @python3 -c "\
import zipfile, glob, sys;\
z = zipfile.ZipFile(glob.glob('dist/*.whl')[0]);\
names = z.namelist();\
plugins = [n for n in names if 'plugins/colormatch/' in n];\
assert len(plugins) == 2, f'expected exactly 2 plugin entries, got {plugins}';\
wheel_meta = z.read([n for n in names if n.endswith('WHEEL')][0]).decode();\
assert 'Tag: py3-none-any' not in wheel_meta, 'wheel is tagged any but contains a native library';\
assert 'Root-Is-Purelib: false' in wheel_meta, 'a wheel with a shared library is not pure Python';\
print('wheel layout ok:', sorted(plugins));\
"

# --- Cross-platform packaging -------------------------------------------------
#
# These names are *targets*, not the host: a wheel is built for a platform, and cross-compilation is
# how the other platforms are produced from one machine. Each target needs its toolchain once:
#   rustup target add aarch64-apple-darwin x86_64-apple-darwin \
#                     x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu \
#                     x86_64-pc-windows-msvc
#   cargo install cargo-zigbuild cargo-xwin
#   brew install zig just
#
# There is no CI yet, so these are run by hand and the artifacts are uploaded manually.

# macOS universal2 wheel (arm64 + x86_64).
wheel-macos-universal:
    cargo build --release -p colormatch-vs --target aarch64-apple-darwin
    cargo build --release -p colormatch-vs --target x86_64-apple-darwin
    lipo -create \
        target/aarch64-apple-darwin/release/libcolormatch.dylib \
        target/x86_64-apple-darwin/release/libcolormatch.dylib \
        -output target/release/libcolormatch.dylib
    cd python && uv build

# Linux x86_64 wheel (manylinux_2_28).
wheel-linux-x64:
    cargo zigbuild --release -p colormatch-vs --target x86_64-unknown-linux-gnu.2.28
    CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu cd python && uv build

# Linux aarch64 wheel (manylinux_2_28).
wheel-linux-arm64:
    cargo zigbuild --release -p colormatch-vs --target aarch64-unknown-linux-gnu.2.28
    CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu cd python && uv build

# Windows x64 wheel.
wheel-windows-x64:
    cargo xwin build --release -p colormatch-vs --target x86_64-pc-windows-msvc
    CARGO_BUILD_TARGET=x86_64-pc-windows-msvc cd python && uv build

# Plain release zip for manual installs / vsrepo-style distribution.
dist-zip: plugin
    mkdir -p dist/colormatch
    cp {{if os() == "macos" { plugin_darwin } else if os() == "windows" { plugin_windows } else { plugin_linux }}} dist/colormatch/
    printf '[VapourSynth Manifest V1]\nlibcolormatch\n' > dist/colormatch/manifest.vs
    cd dist && zip -r colormatch-0.1.0-$(os)-$(arch).zip colormatch

# Remove build output.
clean:
    cargo clean
    rm -rf python/build python/dist dist

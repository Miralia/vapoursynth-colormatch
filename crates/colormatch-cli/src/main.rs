//! `colormatch` — command-line front end.
//!
//! Deliberately free of any VapourSynth dependency so that models can be inspected, exported and
//! (later) built outside a VapourSynth process.

use anyhow::{Context, Result, bail};
use colormatch_core::model::Model;

const USAGE: &str = "\
colormatch — colour matching model tooling

USAGE:
    colormatch <COMMAND>

COMMANDS:
    version              Print the version and exit
    inspect <model.cm>   Print a summary of a serialized model
    cube <model.cm> <out.cube> [--size N]
                         Export a model as an Adobe .cube 3D LUT
";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "version" | "-V" | "--version" => {
            println!("colormatch {}", colormatch_core::VERSION);
            Ok(())
        }
        "inspect" => {
            let path = args.get(1).context("inspect requires a model path")?;
            inspect(path)
        }
        "cube" => {
            let input = args.get(1).context("cube requires a model path")?;
            let output = args.get(2).context("cube requires an output path")?;
            let size = parse_size(&args[3..])?;
            export_cube(input, output, size)
        }
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print!("{USAGE}");
            std::process::exit(2);
        }
    }
}

fn parse_size(rest: &[String]) -> Result<u32> {
    let mut size = 33u32;
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        if a == "--size" {
            size = it.next().context("--size requires a value")?.parse()?;
        } else if let Some(v) = a.strip_prefix("--size=") {
            size = v.parse()?;
        } else {
            bail!("unexpected argument {a}");
        }
    }
    Ok(size)
}

fn load(path: &str) -> Result<Model> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    Model::from_bytes(&bytes).with_context(|| format!("parsing {path} as a .cm model"))
}

fn inspect(path: &str) -> Result<()> {
    let m = load(path)?;
    let s = &m.stats;
    println!("model        {path}");
    println!(
        "format       .cm v{} ({} bytes on disk)",
        m.version,
        m.serialized_len()
    );
    println!("slices (K)   {}", m.len());
    println!("grid nodes   {}", m.nodes);
    println!("samples      {}", s.samples);
    println!("frames       {}", s.frames);
    println!("directions   {}", s.directions);
    println!("sliced W1    {:.6} -> {:.6}", s.w1_before, s.w1_after);
    if s.w1_before > 0.0 {
        println!(
            "reduction    {:.2}%",
            (1.0 - s.w1_after / s.w1_before) * 100.0
        );
    }
    println!("max gain     {:.2}", s.max_gain);
    println!("build time   {:.1} ms", s.build_ms);
    println!(
        "test space   matrix={} range={} transfer={} primaries={}",
        m.test_space.matrix, m.test_space.range, m.test_space.transfer, m.test_space.primaries
    );
    println!(
        "ref space    matrix={} range={} transfer={} primaries={}",
        m.ref_space.matrix, m.ref_space.range, m.ref_space.transfer, m.ref_space.primaries
    );
    Ok(())
}

fn export_cube(input: &str, output: &str, size: u32) -> Result<()> {
    let m = load(input)?;
    let text = colormatch_core::lut::write_cube(&m, size, Some(input))
        .with_context(|| format!("baking a {size}^3 LUT"))?;
    std::fs::write(output, text).with_context(|| format!("writing {output}"))?;
    println!("wrote {output} ({size}^3)");
    Ok(())
}

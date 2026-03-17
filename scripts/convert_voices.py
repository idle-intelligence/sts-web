"""
Convert PersonaPlex .pt voice preset files to sts-web web format.

Each .pt contains:
  embeddings: [num_frames, 1, 1, hidden_dim] bfloat16
  cache:      [1, num_streams, num_positions]  int64

Output per voice:
  {name}.embeddings.bin  -- raw f32le bytes, shape [num_frames, hidden_dim] flattened
  {name}.cache.json      -- {"num_frames": N, "dim": D, "cache": [[stream0...], [stream1...], ...]}

Usage:
  python scripts/convert_voices.py /path/to/voices/ [--output /path/to/output/]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: torch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)


def inspect(path: Path) -> None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"File: {path.name}")
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    else:
        print(f"  type={type(data).__name__}")


def convert(src: Path, dst_dir: Path) -> None:
    data = torch.load(src, map_location="cpu", weights_only=False)

    if not isinstance(data, dict) or "embeddings" not in data or "cache" not in data:
        raise ValueError(f"{src.name}: expected dict with 'embeddings' and 'cache' keys, got {list(data.keys()) if isinstance(data, dict) else type(data)}")

    emb = data["embeddings"]   # [num_frames, 1, 1, hidden_dim] bfloat16
    cache = data["cache"]      # [1, num_streams, num_positions]  int64

    # Squeeze extra dims: [num_frames, hidden_dim]
    emb = emb.squeeze()        # removes all size-1 dims
    if emb.dim() != 2:
        raise ValueError(f"{src.name}: embeddings squeezed to unexpected shape {list(emb.shape)}, expected 2D")

    num_frames, hidden_dim = emb.shape

    # Convert to f32 and pack as little-endian bytes
    emb_f32 = emb.to(torch.float32)
    emb_bytes = struct.pack(f"<{emb_f32.numel()}f", *emb_f32.flatten().tolist())

    # Cache: [1, num_streams, num_positions] -> [num_streams, num_positions]
    cache = cache.squeeze(0)   # [num_streams, num_positions]
    num_streams, num_positions = cache.shape

    cache_lists = cache.tolist()  # list of lists, already int

    name = src.stem
    bin_path = dst_dir / f"{name}.embeddings.bin"
    json_path = dst_dir / f"{name}.cache.json"

    bin_path.write_bytes(emb_bytes)

    cache_obj = {
        "num_frames": num_frames,
        "dim": hidden_dim,
        "cache": cache_lists,
    }
    json_path.write_text(json.dumps(cache_obj, separators=(",", ":")))

    print(f"  {name}: embeddings [{num_frames} × {hidden_dim}] → {bin_path.name} ({len(emb_bytes)} bytes)")
    print(f"  {name}: cache [{num_streams} streams × {num_positions} positions] → {json_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PersonaPlex .pt voice presets to sts-web format")
    parser.add_argument("voices_dir", type=Path, help="Directory containing .pt voice files")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory (default: same as voices_dir)")
    parser.add_argument("--inspect", action="store_true", help="Just print structure of each .pt file, don't convert")
    args = parser.parse_args()

    voices_dir = args.voices_dir.resolve()
    if not voices_dir.is_dir():
        print(f"Error: {voices_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pt_files = sorted(voices_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {voices_dir}", file=sys.stderr)
        sys.exit(1)

    if args.inspect:
        for pt in pt_files:
            inspect(pt)
        return

    dst_dir = (args.output or voices_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(pt_files)} voice(s) from {voices_dir}")
    print(f"Output: {dst_dir}")
    print()

    errors = []
    for pt in pt_files:
        try:
            convert(pt, dst_dir)
        except Exception as e:
            print(f"  ERROR {pt.name}: {e}", file=sys.stderr)
            errors.append(pt.name)

    print()
    if errors:
        print(f"Failed: {errors}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Done. Converted {len(pt_files)} voice preset(s).")


if __name__ == "__main__":
    main()

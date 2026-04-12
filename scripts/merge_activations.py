"""Merge per-batch activation files into per-timestep chunk files.

Converts many small timestep_XXXX_batch_YYY.pt files into fewer, larger
timestep_XXXX_chunk_ZZ.pt files by concatenating CHUNK_SIZE batches at a time.

This reduces file count dramatically (e.g. 6250 → ~120 per layer) while
keeping memory usage bounded.

Old per-batch files and stale index files are removed after merging.
"""

import gc
import re
import sys
from pathlib import Path

import torch

CHUNK_SIZE = 50  # number of batch files to merge into one chunk


def merge_layer(layer_dir: Path) -> None:
    pattern = re.compile(r"timestep_(\d+)_batch_(\d+)\.pt")
    # Group files by timestep
    by_timestep: dict[int, list[tuple[int, Path]]] = {}
    for f in layer_dir.glob("timestep_*_batch_*.pt"):
        m = pattern.match(f.name)
        if not m:
            continue
        ts, batch = int(m.group(1)), int(m.group(2))
        by_timestep.setdefault(ts, []).append((batch, f))

    if not by_timestep:
        print(f"  {layer_dir.name}: no per-batch files found, skipping", flush=True)
        return

    for ts in sorted(by_timestep):
        files = sorted(by_timestep[ts], key=lambda x: x[0])
        total_merged = 0
        chunk_idx = 0

        for i in range(0, len(files), CHUNK_SIZE):
            group = files[i : i + CHUNK_SIZE]
            chunks = []
            for _, f in group:
                chunks.append(torch.load(f, map_location="cpu", weights_only=True))
            combined = torch.cat(chunks, dim=0)
            del chunks
            gc.collect()

            out_path = layer_dir / f"timestep_{ts:04d}_chunk_{chunk_idx:02d}.pt"
            torch.save(combined, out_path)
            total_merged += len(group)
            del combined
            gc.collect()
            chunk_idx += 1

        # Remove old per-batch files
        for _, f in files:
            f.unlink()

        print(
            f"  {layer_dir.name}: timestep {ts} — merged {total_merged} batch files → {chunk_idx} chunks",
            flush=True,
        )

    # Remove stale index file
    index_file = layer_dir / "_file_index.json"
    if index_file.exists():
        index_file.unlink()
        print(f"  {layer_dir.name}: removed stale _file_index.json", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python merge_activations.py <activations_dir>", flush=True)
        sys.exit(1)

    base = Path(sys.argv[1])
    if not base.exists():
        print(f"Directory not found: {base}", flush=True)
        sys.exit(1)

    layer_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("layer_"))
    print(f"Found {len(layer_dirs)} layer directories in {base}", flush=True)

    for layer_dir in layer_dirs:
        merge_layer(layer_dir)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()

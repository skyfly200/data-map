import argparse
import os
from pathlib import Path

import numpy as np
import rasterio

try:
    from rio_cogeo.cogeo import cog_translate, cog_validate
    from rio_cogeo.profiles import cog_profiles
except ImportError:
    cog_translate = None
    cog_validate = None
    cog_profiles = None


def convert_raster_to_cog(input_path, output_path=None, *, delete_original=False, verify=False):
    """Convert a GeoTIFF to a Cloud Optimized GeoTIFF (COG)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if cog_translate is None:
        raise RuntimeError("rio-cogeo is required. Install it with: pip install rio-cogeo")

    in_path = Path(input_path)
    if output_path is None:
        output_path = in_path.with_name(f"{in_path.stem}.cog.tif")
        
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dst_kwargs = cog_profiles.get("deflate") or {
        'COMPRESS': 'DEFLATE',
        'BLOCKSIZE': 512,
        'PREDICTOR': 2,
    }
    
    with rasterio.open(in_path) as src:
        nodata = src.nodata
        min_dim = min(src.width, src.height)
        src_shape = (src.count, src.height, src.width)
        src_dtypes = src.dtypes

    overview_level = 5
    if min_dim > 0:
        max_overview = max(0, int(np.floor(np.log2(min_dim))) - 1)
        overview_level = min(overview_level, max_overview) if max_overview > 0 else 0

    cog_translate(
        in_path,
        str(out_path),
        dst_kwargs=dst_kwargs,
        nodata=nodata,
        overview_level=overview_level or None,
        overview_resampling='average',
        quiet=True,
    )

    if verify:
        with rasterio.open(out_path) as src:
            if src.count < 1:
                raise ValueError(f"COG output is empty: {out_path}")
            
            out_shape = (src.count, src.height, src.width)
            if src_shape != out_shape:
                raise ValueError(f"COG verification failed: shape mismatch for {out_path}")
            if src_dtypes != src.dtypes:
                raise ValueError(f"COG verification failed: dtype mismatch for {out_path}")

    if delete_original and out_path.exists() and in_path != out_path:
        in_path.unlink(missing_ok=True)

    return str(out_path)


def _fmt_size(num_bytes):
    mb = num_bytes / 1e6
    return f"{mb / 1000:.2f} GB" if mb >= 1000 else f"{mb:.1f} MB"


def convert_all_rasters_in_dir(directory, *, replace_originals=False):
    directory = Path(directory)
    
    # Collect all .tif and .tiff files. Do not include files that already end with .cog.tif.
    targets = []
    for ext in ("*.tif", "*.tiff"):
        targets.extend(p for p in directory.rglob(ext) if not p.name.endswith(".cog.tif"))
    targets = sorted(targets)
    
    total = len(targets)
    print(f"🗜  Compressing {total} raster(s) in {directory} to COG...")

    converted = []
    failed = 0
    bytes_before = bytes_after = 0
    
    for i, path in enumerate(targets, 1):
        prefix = f"[{i}/{total}]"
        orig_size = path.stat().st_size if path.exists() else 0
        
        try:
            # Inspect the file. Check if the file is already a valid COG.
            is_valid, _, _ = cog_validate(str(path), quiet=True)
            
            if is_valid:
                new_name = f"{path.stem}.cog.tif"
                new_path = path.with_name(new_name)
                
                if new_path != path:
                    path.rename(new_path)
                    
                print(f"{prefix} ⏭️  {path.name} is already a valid COG. Renamed to {new_path.name}.")
                continue
                
            output = convert_raster_to_cog(path, delete_original=False, verify=True)
            final_path = Path(output)
            
            if replace_originals:
                if final_path.exists() and path.exists() and final_path != path:
                    path.unlink(missing_ok=True)
            
            new_size = final_path.stat().st_size if final_path.exists() else 0
            bytes_before += orig_size
            bytes_after += new_size
            pct = (1 - new_size / orig_size) * 100 if orig_size else 0
            print(f"{prefix} ✅ {path.name}: {_fmt_size(orig_size)} → {_fmt_size(new_size)} "
                  f"({pct:.0f}% smaller)")
            converted.append(output)
            
        except Exception as exc:
            failed += 1
            print(f"{prefix} [!] Failed to process {path}: {exc}")

    saved = bytes_before - bytes_after
    pct = (saved / bytes_before) * 100 if bytes_before else 0
    print(f"🗜  Done — {len(converted)} converted, {failed} failed. "
          f"{_fmt_size(bytes_before)} → {_fmt_size(bytes_after)} "
          f"(saved {_fmt_size(saved)}, {pct:.0f}%).")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert GeoTIFF rasters to COG-compressed TIFFs")
    parser.add_argument("path", help="Raster file or directory of .tif or .tiff files")
    parser.add_argument("--output", default=None, help="Optional output path for a single file")
    parser.add_argument("--delete-original", action="store_true", help="Delete the original uncompressed raster after successful conversion")
    parser.add_argument("--verify", action="store_true", help="Read back the converted raster for a quick validation")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        convert_all_rasters_in_dir(target, replace_originals=args.delete_original)
        return

    result = convert_raster_to_cog(str(target), output_path=args.output, delete_original=args.delete_original, verify=args.verify)
    print(f"COG output: {result}")


if __name__ == "__main__":
    main()
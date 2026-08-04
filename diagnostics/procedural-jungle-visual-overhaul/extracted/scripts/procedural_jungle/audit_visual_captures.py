from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image, ImageStat


def fraction(predicate_values: list[bool]) -> float:
    return sum(1 for value in predicate_values if value) / max(1, len(predicate_values))


def analyse(path: Path) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        width, height = rgb.size
        sample = rgb.resize((320, 180), Image.Resampling.BILINEAR)
        pixels = list(sample.getdata())
        luminance = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels]
        saturation = [(max(r, g, b) - min(r, g, b)) / max(1, max(r, g, b)) for r, g, b in pixels]
        green_dominant = [g > r * 1.04 and g > b * 0.92 and g > 35 for r, g, b in pixels]
        near_white = [r > 238 and g > 238 and b > 238 for r, g, b in pixels]
        near_black = [r < 14 and g < 14 and b < 14 for r, g, b in pixels]
        cyan_water = [b > r * 1.15 and g > r * 1.12 and b > 45 for r, g, b in pixels]
        stat = ImageStat.Stat(sample)
        return {
            'file': path.name,
            'sha256': digest,
            'bytes': len(raw),
            'width': width,
            'height': height,
            'mean_rgb': [round(value, 3) for value in stat.mean],
            'mean_luminance': round(statistics.fmean(luminance), 3),
            'luminance_stdev': round(statistics.pstdev(luminance), 3),
            'mean_saturation': round(statistics.fmean(saturation), 5),
            'green_dominant_fraction': round(fraction(green_dominant), 5),
            'near_white_fraction': round(fraction(near_white), 5),
            'near_black_fraction': round(fraction(near_black), 5),
            'water_colour_fraction': round(fraction(cyan_water), 5),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.input_dir)
    captures = sorted(root.glob('capture_*.png'))
    records = [analyse(path) for path in captures]
    errors: list[str] = []
    if len(records) != 8:
        errors.append(f'expected exactly 8 captures, found {len(records)}')
    if len({record['sha256'] for record in records}) != len(records):
        errors.append('captures are not byte-distinct')
    for record in records:
        if (record['width'], record['height']) != (1920, 1080):
            errors.append(f"{record['file']} is not 1920x1080")
        if record['bytes'] < 100_000:
            errors.append(f"{record['file']} is implausibly small")
        if not 22.0 <= record['mean_luminance'] <= 205.0:
            errors.append(f"{record['file']} luminance outside controlled range: {record['mean_luminance']}")
        if record['luminance_stdev'] < 20.0:
            errors.append(f"{record['file']} lacks tonal depth: {record['luminance_stdev']}")
        if record['near_white_fraction'] > 0.28:
            errors.append(f"{record['file']} is washed out: {record['near_white_fraction']}")
        if record['near_black_fraction'] > 0.45:
            errors.append(f"{record['file']} is excessively black: {record['near_black_fraction']}")

    average_green = statistics.fmean(record['green_dominant_fraction'] for record in records) if records else 0.0
    average_saturation = statistics.fmean(record['mean_saturation'] for record in records) if records else 0.0
    average_white = statistics.fmean(record['near_white_fraction'] for record in records) if records else 1.0
    rich_green_frames = sum(record['green_dominant_fraction'] >= 0.20 for record in records)
    water_frames = sum(record['water_colour_fraction'] >= 0.015 for record in records)
    if average_green < 0.25:
        errors.append(f'average green coverage below jungle threshold: {average_green:.5f}')
    if rich_green_frames < 6:
        errors.append(f'fewer than six vegetation-rich frames: {rich_green_frames}')
    if average_saturation < 0.19:
        errors.append(f'average saturation below threshold: {average_saturation:.5f}')
    if average_white > 0.16:
        errors.append(f'average near-white coverage too high: {average_white:.5f}')
    if water_frames < 2:
        errors.append(f'fewer than two frames visibly contain water: {water_frames}')

    payload = {
        'classification': 'DENSE_JUNGLE_VISUAL_QUALITY_PROVEN' if not errors else 'REJECTED',
        'errors': errors,
        'capture_count': len(records),
        'unique_capture_count': len({record['sha256'] for record in records}),
        'average_green_dominant_fraction': round(average_green, 5),
        'average_saturation': round(average_saturation, 5),
        'average_near_white_fraction': round(average_white, 5),
        'vegetation_rich_frame_count': rich_green_frames,
        'water_frame_count': water_frames,
        'captures': records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
from pathlib import Path

EXPECTED = {
    "JungleProofDirector.h": "c7bc7fd670692155a61b2293a9a2fbb5901ff9e0670de348931180b2ad14949f",
    "JungleProofDirector.cpp": "6c93a7d01d3a90dc64d2f636d87bc22e79694096efaf0397903ef2d411a9a3d8",
}
REPLACEMENTS = {
    "JungleProofDirector.h": "H4sIAJAYcmoC/7VUXU/bMBR9z6+wijS1CDH2gYbKhxTSlIEYREk6xF6Qm9ymXh07c5xCNe2/78ZxS6DhbetD1fqce+71vcd3p1A0yymRIgHH2WEi4VUKpOdJBd+YYDnl+/NeC7igOYwVfj1KtXjvJlqql4SrSmQcAiXlbMQUGEIGAhTVkNZUJ+G0LInroYiiRuF4fRZQkdI7yhegvDlVNNGwASdRgjIeLXSlwJN5IQUI/XG0wWN4qqEQRAoqpioDgzoT79qNov7A8oLw1vNHk9C9vprcXFz7h0cPbnBJ3I7CyZAU1ZSzhLimTue3Q/Bz4d/4oRv7o4fz29E9CjekoQG7dPqDY4MtmdIV5WQpWUrOIWMi4HTVHxC5BKVYCh20mCWL/oxLqskIuKYRJFKkZTvGBE3wXoEfxvd9P2XaFavHOSjYI+e8gkIxoUOg6Z1iGs88nEUm1eq0Z8rsDYxCk8Q2+FJg65dYxCn5sP/lcPZ/skSgNQe8F11hooP9zwf/LhFyPn0kIfyqcArpxjeV0JjqCNOgzBIjh6/yNdHx7fQnzi7Q6uQt452RNoCiouK80GrrBtuKnVY9W3elOdgSjF2l6Oqk/XB265j6X9k4p/MB7RJz2tarueOYYeRXhDiQQEFBMXN91AGvC3uGm+5uzJLCUz3ADszMaEwZb3X/Bc8UZ1bK6yFZWuMYn9OihNTYZNYGmldR5duIOy0lr3SzyKp8XJS1mw/ws+Y1FZjkEc0LDuVzWttuI3VGMNgybHu+NxvCVB9pquqC7eFw+AOUbH5bdqTRsxkx/gyl1Hak5oXb3t+gJ2wDNvvCLAC6AEtJu/BAlkwzKUwljRv6LzxiLbLXui3WYmZmdaZS8vW8GnIsA5G9IYP7p9TrK70jY8ZBINSuaYwdL+fmtnWtf46dvxWA0kRpBgAA",
    "JungleProofDirector.cpp": "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",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    source_dir = Path(args.root) / "project_template" / "Source" / "ProceduralJungle58"
    for name, expected_hash in EXPECTED.items():
        path = source_dir / name
        actual_hash = sha256(path.read_bytes())
        if actual_hash != expected_hash:
            raise SystemExit(
                f"{name} source hash mismatch: expected={expected_hash} actual={actual_hash}"
            )

    for name, encoded in REPLACEMENTS.items():
        path = source_dir / name
        replacement = gzip.decompress(base64.b64decode(encoded))
        path.write_bytes(replacement)
        print(
            f"JUNGLE_OFFSCREEN_CAPTURE_SOURCE_PATCH={name};SHA256={sha256(replacement)}",
            flush=True,
        )

    combined = "\n".join(
        (source_dir / name).read_text(encoding="utf-8") for name in REPLACEMENTS
    )
    required = [
        "USceneCaptureComponent2D",
        "UKismetRenderingLibrary::CreateRenderTarget2D",
        "UKismetRenderingLibrary::ReadRenderTarget",
        "UKismetRenderingLibrary::ExportRenderTarget",
        "JUNGLE_OFFSCREEN_CAPTURE_NONBLACK_FRACTION",
        "SceneCapture2D_RenderTarget_PNG",
    ]
    missing = [marker for marker in required if marker not in combined]
    if missing:
        raise SystemExit(f"off-screen capture markers missing: {missing}")
    print("JUNGLE_OFFSCREEN_CAPTURE_SOURCE_PATCH=PROVEN", flush=True)


if __name__ == "__main__":
    main()

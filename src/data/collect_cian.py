"""Collect fresh apartment listings from CIAN.

Example:
    python -m src.data.collect_cian --location "Санкт-Петербург" --start-page 1 --end-page 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_rooms(value: str) -> tuple[int | str, ...] | str:
    """Parse CLI rooms argument into a cianparser-compatible value."""
    value = value.strip().lower()
    if value == "all":
        return "all"

    rooms: list[int | str] = []
    for part in value.split(","):
        item = part.strip().lower()
        if item in {"studio", "студия"}:
            rooms.append("studio")
        else:
            rooms.append(int(item))
    return tuple(rooms)


def collect_flats(
    location: str,
    start_page: int,
    end_page: int,
    rooms: tuple[int | str, ...] | str,
    output_dir: Path,
    with_extra_data: bool,
    timeout: int,
    proxy: str | None,
) -> Path:
    """Collect CIAN sale listings and save them as CSV."""
    try:
        import cianparser
    except ImportError as exc:
        raise SystemExit(
            "cianparser is not installed. Run: pip install -r requirements.txt"
        ) from exc

    parser = cianparser.CianParser(location=location, proxies=[proxy] if proxy else None)
    parser.__session__.headers.update({
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    })
    original_get = parser.__session__.get
    parser.__session__.get = lambda *args, **kwargs: original_get(
        *args,
        timeout=kwargs.pop("timeout", timeout),
        **kwargs,
    )

    data = parser.get_flats(
        deal_type="sale",
        rooms=rooms,
        with_saving_csv=False,
        with_extra_data=with_extra_data,
        additional_settings={
            "start_page": start_page,
            "end_page": end_page,
        },
    )

    df = pd.DataFrame(data)
    collected_at = datetime.now(timezone.utc).isoformat()
    df["source"] = "cian"
    df["collected_at"] = collected_at
    df["location_query"] = location
    df["deal_type"] = "sale"

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_location = location.lower().replace(" ", "_").replace("-", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cian_{safe_location}_sale_p{start_page}_{end_page}_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    print(f"Collected {len(df)} listings")
    print(f"Saved to {output_path}")
    if len(df) > 0:
        print("Columns:")
        print(", ".join(df.columns.astype(str)))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect apartment sale listings from CIAN.")
    parser.add_argument("--location", default="Санкт-Петербург", help="CIAN location name.")
    parser.add_argument("--start-page", type=int, default=1, help="First search results page.")
    parser.add_argument("--end-page", type=int, default=2, help="Last search results page.")
    parser.add_argument(
        "--rooms",
        default="1,2",
        help='Rooms to collect: "1,2", "studio,1", or "all".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where the CSV snapshot will be saved.",
    )
    parser.add_argument(
        "--with-extra-data",
        action="store_true",
        help="Collect extra listing details. Slower, but useful for richer features.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Request timeout in seconds. Prevents hanging on blocked or slow pages.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optional HTTPS proxy, e.g. http://user:pass@host:port.",
    )
    args = parser.parse_args()

    collect_flats(
        location=args.location,
        start_page=args.start_page,
        end_page=args.end_page,
        rooms=parse_rooms(args.rooms),
        output_dir=args.output_dir,
        with_extra_data=args.with_extra_data,
        timeout=args.timeout,
        proxy=args.proxy,
    )


if __name__ == "__main__":
    main()

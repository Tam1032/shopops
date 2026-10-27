"""Load a public review dataset (CSV) into the mock marketplace.

Works with any CSV that has a text column and, optionally, a rating column,
e.g. UIT-ViSFD or the Shopee/Tiki sentiment sets on Kaggle. Each review is
attached to a random product and gets its own delivered order.

    python scripts/import_reviews.py path/to/reviews.csv --text comment --rating n_star --limit 300

Imported reviews have no labels. To grow the golden set, label a sample by hand
and append it to evals/golden.jsonl in the same format.
"""

import argparse
import csv
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--text", default="comment")
    p.add_argument("--rating", default=None, help="rating column (1-5); missing means 3")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    catalog = json.loads((DATA / "catalog.json").read_text(encoding="utf-8"))
    skus = [prod["sku"] for prod in catalog["products"]]
    reviews_path = DATA / "reviews.jsonl"
    existing = [json.loads(line) for line in reviews_path.read_text(encoding="utf-8").splitlines() if line]
    next_id = len(existing) + 1

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get(args.text) or "").strip()]
    rng.shuffle(rows)

    new_reviews, new_orders = [], []
    for row in rows[: args.limit]:
        rid, oid = f"R{next_id:04d}", f"O{5000 + next_id}"
        sku = rng.choice(skus)
        variant = next(iter(next(p for p in catalog["products"] if p["sku"] == sku)["variants"]))
        rating = int(float(row[args.rating])) if args.rating and row.get(args.rating) else 3
        new_orders.append({"order_id": oid, "sku": sku, "variant": variant, "qty": 1, "status": "delivered",
                           "carrier": "GHN", "ordered_at": "2026-10-01", "promised_by": "2026-10-04",
                           "delivered_at": "2026-10-03"})
        new_reviews.append({"review_id": rid, "order_id": oid, "sku": sku, "rating": max(1, min(5, rating)),
                            "text": row[args.text].strip(), "channel": rng.choice(["shopee", "tiktok"]),
                            "created_at": "2026-10-20T09:00:00"})
        next_id += 1

    catalog["orders"] += new_orders
    (DATA / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    with reviews_path.open("a", encoding="utf-8") as f:
        for r in new_reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"added {len(new_reviews)} reviews")


if __name__ == "__main__":
    main()

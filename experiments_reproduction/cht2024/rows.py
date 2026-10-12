"""Table 1's rows.

Each row is (paper label, config overrides), labelled exactly as the paper
prints it so the report can put the paper's number next to ours.
"""


def table1_rows(scale):
    """FC, RigL, MEST and CHT (from ER, i.e. WS beta=1) on the CNN heads."""
    s = f"{scale}×"
    return [
        (f"FC{s}", dict(method="fc")),
        (f"RigL{s}", dict(method="rigl")),
        (f"MEST{s}", dict(method="mest")),
        (f"CHT{s}", dict(method="cht", init="er")),
    ]


def select(rows, patterns):
    """Keep rows whose label contains any of `patterns` (all rows if none given)."""
    if not patterns:
        return rows
    return [r for r in rows if any(p in r[0] for p in patterns)]

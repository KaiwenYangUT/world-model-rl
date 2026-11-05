"""CSV schemas (PLAN.md §7) and a writer that enforces them."""

import csv
import os

EVAL_COLUMNS = (
    "exp", "method", "seed", "ckpt_env_steps", "wall_clock_s", "split", "n_episodes",
    "success_rate", "success_ci_lo", "success_ci_hi", "mean_return", "mean_len",
    "mean_steps_to_success", "target", "reward_mode")
WM_ERROR_COLUMNS = (
    "exp", "seed", "k", "n", "rew_mae", "rew_mae_event", "rew_event_recall",
    "rew_event_precision", "cont_brier", "cont_bce", "term_recall", "cum_rew_err")
SCHEMAS = {"eval": EVAL_COLUMNS, "wm_error": WM_ERROR_COLUMNS}


def write_rows(path, rows, columns, append=False):
    """Write dict rows with exactly `columns` (a schema name or a column list).

    Every row must have exactly these keys, so schema drift fails loudly. With append=True,
    an existing file must already have the same header; the header is written only once.
    None is written as an empty cell.
    """
    columns = list(SCHEMAS[columns] if isinstance(columns, str) else columns)
    for row in rows:
        if set(row) != set(columns):
            missing, extra = set(columns) - set(row), set(row) - set(columns)
            raise ValueError(f"row does not match schema: missing {sorted(missing)}, "
                             f"extra {sorted(extra)}")
    exists = append and os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if header != columns:
            raise ValueError(f"{path}: header {header} does not match schema {columns}")
    with open(path, "a" if exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

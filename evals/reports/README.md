# Evaluation reports

This is the canonical, tracked output directory for take-home evaluation runs. It preserves
successful and failed experiments so reviewers can distinguish measured results from attempted
runs. Reports must contain the dataset digest, code and prompt/retrieval versions, profile,
protocol, completion status, and aggregate/per-case metrics when available.

Reports must never contain credentials, environment dumps, raw database contents, private trace
links, or provider error bodies. A failed run should record only a stable error code and exception
class. The CLIs fail before replacing an existing single report or starting a new matrix in a
non-empty label directory. `--resume` extends only a compatible interrupted matrix. Use a distinct
run label or timestamped directory for every new experiment.

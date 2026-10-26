# Experiments log

One row per experiment or audit run. Metrics are on HELD-OUT replay files unless stated.

| Date | Change / run | Metric (held-out) | Verdict |
|------|--------------|-------------------|---------|
| 2026-10-26 | Audit of old yt2slp corpora + models (no training). Scripts: `audit_corpus_stats.py`, `check_label_alignment.py --all`, `summarize_old_models.py`, `scan_slp_library.py`. See `docs/audit-old-assets.md`. | Label vs .slp mismatches: 0. HUD-percent onset offset over 1,090 hits: 0 frames for 94.4%, -1 for 5.6% (3 inspected = detector artifacts); timer: no off-by-one in 284 clips. Old controller model: stick-X acc 0.478 vs 0.456 always-neutral; press F1 0.10 (old project's validation split). | Old clips are correctly labeled and frame-aligned but tiny and narrow; old models not useful. Re-render from SLP library. |
| 2026-10-26 | Milestone 1 data pipeline: index (`build_slp_index.py`), opponent-grouped split seed 0 (`make_splits.py`), peppi-py parse to Parquet with rollback de-dup (`parse_slps.py`). No training. | 3,435 eligible games parsed, 0 failures, 0 missing frames; 58.9M player-frame rows, 1.09 GiB. Split 2,738 / 353 / 344 games (704 / 160 / 146 opponents), 0 leaks. Parser matches verified old labels exactly (float32) on 216,000 values. 49 library files unreadable by peppi-py 0.8.6 (unknown game-end code). | Done. Ready for milestone 2. |

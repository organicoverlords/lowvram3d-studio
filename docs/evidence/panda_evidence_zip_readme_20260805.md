# Panda evidence ZIP

This archive contains the compact visual and receipt evidence for the 2026-08-05 panda front-projection check.

- `golden_baseline/` is the preserved historical generated-multiview visual baseline.
- `rejected_highres_projection/` is the sharper-source direct-front diagnostic rejected by manual review.
- `receipts/` contains the projection, render, inference and rejection reports.
- Large GLBs and provenance NPZ arrays are intentionally not embedded; their exact paths and SHA256 values are recorded in the report.

The correct classification is: golden baseline preserved; high-resolution direct-front projection rejected; bottom white-invalid region remains; full 360 production is not proven.

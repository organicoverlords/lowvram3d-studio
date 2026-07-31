# Installer fix 0.5.1

- Reads Windows-authored JSON using `utf-8-sig` so existing BOM-prefixed discovery/config files resume safely.
- Writes new installer JSON without a UTF-8 BOM.
- Corrects TripoSR checkout validation to require `tsr/system.py` rather than a nonexistent `tsr/__init__.py`.
- Bumps the optional TripoSR checkpoint fingerprint so the corrected stage is retried once automatically.
- Preserves all verified checkpoints from v0.5.0.

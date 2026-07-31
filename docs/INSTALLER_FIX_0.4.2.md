# Installer fix 0.4.2

The 0.4.1 installer reached the MV-Adapter environment successfully, but the
upstream legacy editable build read package metadata through the Windows cp1252
codec and failed with `UnicodeDecodeError`.

Version 0.4.2 does not run `pip install -e` for MV-Adapter. The worker already
adds the pinned repository root to `sys.path`, so the editable build is redundant.
The installer now:

1. enables Python UTF-8 mode;
2. writes a UTF-8 `.pth` file pointing at the pinned MV-Adapter repository;
3. verifies `import mvadapter` against that repository;
4. resumes with the existing downloaded Torch and dependency environment.

Run `CONTINUE-INSTALL.cmd`; no uninstall or cleanup is required.

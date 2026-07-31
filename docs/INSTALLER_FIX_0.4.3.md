# Installer fix 0.4.3

## Failure

TripoSR's `torchmcubes` dependency attempted a local CMake/C++/CUDA build on
Windows. Runtime-only PyTorch installations do not expose a complete native
build toolchain, so CMake failed before the optional emergency lane could be
registered.

## Fix

- Removed the native `torchmcubes` build from the one-click installer.
- Added a compatible `torchmcubes.marching_cubes` module backed by
  `skimage.measure.marching_cubes` on CPU.
- Added a synthetic sphere extraction test during installation.
- Pinned NumPy 1.26.4 and scikit-image 0.24.0 in the TripoSR environment.
- Made Lane C non-blocking: a TripoSR-specific failure is recorded and reported,
  while the primary Mini Turbo and deterministic Blender lanes remain usable.

The compatibility layer is only used for emergency geometry extraction. Neural
inference still uses the installed CUDA-enabled PyTorch environment.

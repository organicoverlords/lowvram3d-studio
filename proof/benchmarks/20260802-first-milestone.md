# Six-benchmark first milestone — character baseline evidence

The following two assets have actual non-empty textured GLBs and direct fresh-process Blender
proof. The generated artifacts remain in the external benchmark run root so the repository does
not absorb large binary outputs.

## tactical_red_panda_scout — original route REJECTED; repaired route PROVEN

The original Smart-UV/textured export is explicitly rejected. It produced a disconnected
triangle soup and a corrupted atlas, so it is not an accepted image-to-model result:

- Rejected GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured\tactical_red_panda_scout_textured.glb`
- Rejection: `GEOMETRY_BASELINE=REJECTED_DISCONNECTED_TRIANGLE_SOUP`; `UV_OR_COORDINATE_ROUTE=REJECTED_EXCESSIVE_PER_TRIANGLE_SPLITTING`; `BASE_COLOR_TEXTURE=REJECTED_CORRUPTED_ATLAS`.

The repaired route uses the existing raw geometry, a shared planar-front UV projection, and a
targeted cleanup of the proven detached rod/blob artifact. No new GPU generation was run.

- Isolated artifact proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\diagnostics\component_140_renders\preview_front.png`

- Source: `C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems\benchmarks\source-images\20260802-six-variation-pack\02_tactical_red_panda_scout.jpg`
- Conditioning audit: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\normalized_conditioning_audit.json`
- Generation reports: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\attempt1_report.json`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\attempt2_report.json`
- Geometry GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\geometry_attempt2.glb`
- Textured GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured_repaired_v7\tactical_red_panda_scout_repaired_v7.glb`
- Base colour: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured_repaired_v7\tactical_red_panda_scout_repaired_v7_basecolor.png`
- Fresh import/material proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured_repaired_v7\fresh_import_validation.json`
- Geometry cleanup proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured_repaired_v7\work\geometry_cleanup_report.json`
- Raster proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured_repaired_v7\work\raster_report.json`
- Visible renders/contact sheet: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\repaired_v7\preview_front.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\repaired_v7\preview_three_quarter.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\repaired_v7\preview_side.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\repaired_v7\contact_sheet.png`
- SHA-256: GLB `4A7E93EAC1DFD9F705AB682DD33051CEA1A2D28DD5A5E227B72C949425803B0C`; base colour `0DF3C85567C1F6A0471657949EDDA2AB4BAE5BF06F0C62150D077902DA6E05E3`
- Repaired geometry: cleanup `667977` faces, one kept welded subject component; final fresh import `667975` polygons; independent indexed check `0.3089%` single-use vertices rather than the rejected `99.70%`.
- Texture evidence: `85.2%` observed semantic coverage, `14.8%` explicitly synthesized local fill for unseen/rear regions, `0` residual island holes; front render preserves face, fur/suit, rifle, and tail colours.
- Generation policy: normalized conditioning; steps 1 failed with the allowed empty-reduction code; one steps 2 retry succeeded; octree 256, chunks 1500, seed 12345 unchanged.

## frog_salvage_diver — PROVEN

- Source: `C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems\benchmarks\source-images\20260802-six-variation-pack\01_frog_salvage_diver.png`
- Conditioning audit: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\logs\normalized_conditioning_audit.json`
- Generation report: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\state\GENERATE\candidate\generate_report.json`
- Geometry GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\state\GENERATE\candidate\master.glb`
- Textured GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\textured\frog_salvage_diver_textured.glb`
- Base colour: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\textured\frog_salvage_diver_textured_basecolor.png`
- Fresh import/material proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\logs\textured_baseline_validation.json`
- Visible renders/contact sheet: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\renders\textured_material\preview_front.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\renders\textured_material\preview_three_quarter.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\renders\textured_material\preview_side.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\frog_salvage_diver\renders\frog_salvage_diver_contact_sheet.png`
- SHA-256: GLB `9B196974D8DC8D70698896CA94E6F077B962EF568FD1A180F1A6A8EC42869B57`; base colour `2DA39AE17017B70B9A1893254A634F10FB782CB3AD9B971F4104DE1984132F7A`
- Generation policy: normalized conditioning; steps 1 failed with the allowed empty-reduction code; one steps 2 retry succeeded; octree 256, chunks 1500, seed 12345 unchanged.

The validation reports independently confirm non-empty geometry, UVs, material presence, packed
readable non-constant base-colour textures linked to Principled Base Color, and rendered previews.
No rigging or animation was run for these static baselines.

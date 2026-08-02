# Six-benchmark first milestone — proven character baselines

The following two assets have actual non-empty textured GLBs and direct fresh-process Blender
proof. The generated artifacts remain in the external benchmark run root so the repository does
not absorb large binary outputs.

## tactical_red_panda_scout — PROVEN

- Source: `C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems\benchmarks\source-images\20260802-six-variation-pack\02_tactical_red_panda_scout.jpg`
- Conditioning audit: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\normalized_conditioning_audit.json`
- Generation reports: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\attempt1_report.json`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\attempt2_report.json`
- Geometry GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\geometry_attempt2.glb`
- Textured GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured\tactical_red_panda_scout_textured.glb`
- Base colour: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\textured\tactical_red_panda_scout_textured_basecolor.png`
- Fresh import/material proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\logs\textured_baseline_validation.json`
- Visible renders/contact sheet: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\textured_material\preview_front.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\textured_material\preview_three_quarter.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\textured_material\preview_side.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802\tactical_red_panda_scout\renders\tactical_red_panda_scout_contact_sheet.png`
- SHA-256: GLB `CCF23D7A474C1CB9D21CA782EA2853FB79EBBE806485DACDDA27F0F72357485A`; base colour `A2EB32578478ECAD7794B3CDB32827B4FCCBA3A4DB941D73D2057F010138666A`
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

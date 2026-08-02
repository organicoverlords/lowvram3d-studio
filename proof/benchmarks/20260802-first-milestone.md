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
- Repaired geometry: cleanup `667977` faces, one kept welded subject component; normal fresh import `667977` polygons; independent indexed check `0.3089%` single-use vertices rather than the rejected `99.70%`. The stricter weld/triangulate checker counted `667975` triangles after export, so `GEOMETRY_QUALITY_EXACT_FACE_COUNT_GATE=NOT_PROVEN` (two-face discrepancy), while the direct import/material/render gate passed.
- Texture evidence: `85.2%` observed semantic coverage, `14.8%` explicitly synthesized local fill for unseen/rear regions, `0` residual island holes; front render preserves face, fur/suit, rifle, and tail colours.
- Generation policy: normalized conditioning; steps 1 failed with the allowed empty-reduction code; one steps 2 retry succeeded; octree 256, chunks 1500, seed 12345 unchanged.

## frog_salvage_diver — PROVEN_WITH_LIMITATIONS

The earlier frog textured artifact is not the accepted result. The bounded fresh run below uses
the current Mini Turbo route, generic relative/source/screen component evidence, the repaired
raster route, and a source-projection view built from the proven transparent matte rather than the
raw white-background image.

- Source: `C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems\benchmarks\source-images\20260802-six-variation-pack\01_frog_salvage_diver.png`
- Conditioning audit/matte: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\state\INGEST\proven\matte.png`
- Generation report: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\state\GENERATE\candidate\generate_report.json`
- Raw geometry: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\state\GENERATE\proven\master.glb`
- Generic component audit: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\geometry\component_audit_creature_partial_confirmed.json`
- UV candidate: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\uv\planar_uv_candidate.glb`
- Textured GLB: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\textured_repaired_v7_matte\frog_salvage_diver_repaired_v7_matte.glb`
- Base colour: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\textured_repaired_v7_matte\frog_salvage_diver_repaired_v7_matte_basecolor.png`
- Fresh import/material proof: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\textured_repaired_v7_matte\fresh_import_validation.json`
- Visible renders/contact sheet: `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\renders\textured_repaired_v7_matte\preview_front.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\renders\textured_repaired_v7_matte\preview_three_quarter.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\renders\textured_repaired_v7_matte\preview_side.png`, `C:\AI\LowVRAM3D-benchmarks\six-benchmark-baselines-20260802-frog-v7\frog_salvage_diver\renders\frog_salvage_diver_repaired_v7_matte_contact_sheet.png`
- SHA-256: GLB `28AB44CDC1CF112EEAA66A86DFE4E8AC262F2F1C2018790AA28AD2DCBBB38F71`; base colour `A2CF3C2DF70CA2C13DDCE120B8151D1A57AFA7CE3A55FA202CBE002E47D4964C`
- Generation policy: normalized conditioning; steps 1 failed with `EXPECTED_REDUCTION_DIM_NON_ZERO`; exactly one steps 2 retry succeeded; octree 256, chunks 1500, seed 12345 unchanged; no fallback generator, rigging, or animation.
- Geometry evidence: generic audit applied only confirmed debris removals (`5780` faces, `0.9159%`) and preserved unresolved components; topology remained `boundary_edges 28 -> 28`, `non_manifold_edges 39 -> 39`; `manual_review_required=true`.
- Texture evidence: `76.85%` observed semantic coverage, `23.15%` constrained synthesized fill, `100%` final filled UVs, and `0` residual island holes; fresh Blender import/material/texture/render checks all passed.
- Visual limitation: source identity is recognizable, but attached generated surface noise and several small floating dark components remain visible in three-quarter/side views. Therefore `DETACHED_ARTIFACT_CLEANUP=NOT_PROVEN_COMPLETE` and this is not a clean production-quality geometry result.

The validation reports independently confirm non-empty geometry, UVs, material presence, packed
readable non-constant base-colour textures linked to Principled Base Color, and rendered previews.
No rigging or animation was run for these static baselines.

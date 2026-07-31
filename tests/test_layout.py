from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_no_implementation_file_exceeds_600_lines(self):
        folders = ["src", "service", "workers", "blender", "scripts", "comfyui_nodes"]
        offenders = []
        for folder in folders:
            for path in (ROOT / folder).rglob("*"):
                if path.suffix.lower() not in {".py", ".ps1", ".js", ".json"} or not path.is_file():
                    continue
                lines = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
                if lines > 600:
                    offenders.append((str(path.relative_to(ROOT)), lines))
        self.assertEqual(offenders, [])

    def test_three_lanes_are_configured(self):
        config = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(config["lane_order"], ["A", "B", "C"])
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("MV-Adapter", installer)
        self.assertIn("TripoSR", installer)
        self.assertIn("3DGenStudio", installer)

    def test_studio_uses_dedicated_non_router_port(self):
        default_config = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        start = (ROOT / "START-STUDIO.ps1").read_text(encoding="utf-8")
        registration = (ROOT / "scripts" / "register_studio.py").read_text(encoding="utf-8")
        self.assertEqual(default_config["studio_url"], "http://127.0.0.1:8311")
        self.assertIn("127.0.0.1:8311", start)
        self.assertIn("127.0.0.1:8311", registration)
        self.assertNotIn("127.0.0.1:3001", start)

    def test_node_version_matches_vite8_runtime_requirement(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("function Test-NodeCompatible", installer)
        self.assertIn("$version.Major -eq 20", installer)
        self.assertIn("$version.Minor -ge 19", installer)
        self.assertIn("$version.Major -eq 22", installer)
        self.assertIn("$version.Minor -ge 12", installer)
        self.assertIn("winget upgrade --id OpenJS.NodeJS.LTS", installer)

    def test_optional_failures_do_not_repeat_on_every_resume(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        checkpoints = (ROOT / "scripts" / "windows" / "install-checkpoints.ps1").read_text(encoding="utf-8")
        self.assertIn("[switch]$RetryOptional", installer)
        self.assertIn("-RetryDegraded:$RetryOptional", installer)
        self.assertIn("status -eq 'degraded'", checkpoints)
        self.assertIn("-Status 'degraded'", checkpoints)
        self.assertTrue((ROOT / "RETRY-OPTIONAL.cmd").is_file())

    def test_uv_pip_uses_flat_argument_vector_and_named_arrays(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("$uvArgs = @('pip', $Arguments[0], '--python', $Python)", installer)
        self.assertIn('& uv @uvArgs', installer)
        self.assertIn("UvPip -Python $ControlPython -Arguments @('install','-r'", installer)
        self.assertIn("UvPip -Python $MeshPython -Arguments @('install','-r'", installer)
        self.assertNotRegex(installer, r"UvPip \$[A-Za-z]+Python @\(")

    def test_uv_pip_calls_do_not_quote_install_and_requirements_as_one_argument(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertNotIn("'install -r ", installer)
        self.assertNotIn('"install -r ', installer)
        for line in installer.splitlines():
            if line.strip().startswith('UvPip '):
                self.assertIn('-Python ', line)
                self.assertIn('-Arguments @(', line)

    def test_mv_adapter_avoids_legacy_editable_build(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertNotIn("@('install','-e',$MvRoot,'--no-deps')", installer)
        self.assertIn("lowvram3d_mv_adapter_repo.pth", installer)
        verifier = (ROOT / "scripts" / "verify_mv_adapter_env.py").read_text(encoding="utf-8")
        self.assertIn("mvadapter", verifier)
        self.assertIn("$env:PYTHONUTF8 = '1'", installer)


    def test_triposr_uses_cpu_marching_cubes_without_native_build(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertNotIn("git+https://github.com/tatsy/torchmcubes.git", installer)
        self.assertIn("install_torchmcubes_cpu_shim.py", installer)
        self.assertIn("scikit-image==0.24.0", installer)
        self.assertIn("TRIPOSR_UNAVAILABLE.txt", installer)
        self.assertIn("triposr_ready", installer)
        shim = (ROOT / "scripts" / "install_torchmcubes_cpu_shim.py").read_text(encoding="utf-8")
        self.assertIn("measure.marching_cubes", shim)
        self.assertIn("CPU marching-cubes verification produced no vertices", shim)

    def test_windows_bom_json_is_supported_and_new_json_is_no_bom(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        config_loader = (ROOT / "src" / "lowvram3d" / "config.py").read_text(encoding="utf-8")
        config_writer = (ROOT / "scripts" / "write_local_config.py").read_text(encoding="utf-8")
        checkpoints = (ROOT / "scripts" / "windows" / "install-checkpoints.ps1").read_text(encoding="utf-8")
        self.assertIn("Write-Utf8NoBom", installer)
        self.assertIn("WriteAllText", installer)
        self.assertIn('encoding="utf-8-sig"', config_loader)
        self.assertIn('encoding="utf-8-sig"', config_writer)
        self.assertIn("Write-JsonUtf8NoBom", checkpoints)

    def test_triposr_checkout_validates_real_namespace_package_files(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("@('tsr\\system.py','run.py')", installer)
        self.assertNotIn("@('tsr\\__init__.py','run.py')", installer)
        self.assertIn("cpu-mc-v3", installer)

    def test_postprocess_is_split_into_focused_module(self):
        pipeline_lines = len((ROOT / "src" / "lowvram3d" / "pipeline.py").read_text(encoding="utf-8").splitlines())
        self.assertLess(pipeline_lines, 650)
        self.assertTrue((ROOT / "src" / "lowvram3d" / "postprocess.py").is_file())

    def test_trellis_is_not_a_backend(self):
        code = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for folder in ("src", "service", "workers") for path in (ROOT / folder).rglob("*.py"))
        self.assertNotIn("TRELLIS", code)

    def test_installer_is_checkpointed_and_resume_first(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        checkpoints = (ROOT / "scripts" / "windows" / "install-checkpoints.ps1").read_text(encoding="utf-8")
        self.assertIn("install-state\\stages", installer)
        self.assertIn("Invoke-InstallStage", installer)
        self.assertIn("[resume] $Name already complete; skipping.", checkpoints)
        self.assertIn("Stage '$Name' action finished but its readiness check did not pass.", checkpoints)
        self.assertIn("status = $Status", checkpoints)
        self.assertIn("fingerprint = $Fingerprint", checkpoints)
        self.assertIn("attempts = $Attempts", checkpoints)
        self.assertIn("Write-InstallSummary", installer)
        self.assertIn("recorded as degraded", checkpoints)
        self.assertIn("RETRY-OPTIONAL.cmd", checkpoints)
        self.assertIn("[switch]$AdoptExisting", checkpoints)
        self.assertIn("-not $checkpoint -and $AdoptExisting", checkpoints)


    def test_version_sensitive_stages_cannot_adopt_stale_proof(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        for stage in (
            "03-local-discovery", "10-local-configuration", "11-runtime-service-smoke",
            "13-comfyui-bridge", "14-package-verification", "15-shortcuts",
        ):
            line = next(line for line in installer.splitlines() if f"-Name '{stage}'" in line)
            self.assertNotIn("-AdoptExisting", line, stage)
        for stage in ("02-control-environment", "04-studio-source", "08-mv-adapter-environment"):
            line = next(line for line in installer.splitlines() if f"-Name '{stage}'" in line)
            self.assertIn("-AdoptExisting", line, stage)

    def test_resume_does_not_repeat_large_completed_installs(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("Test-StudioNodeRuntime $StudioRoot", installer)
        self.assertEqual(installer.count("npm ci --no-audit --no-fund --ignore-scripts"), 1)
        self.assertNotIn("fetch --all", installer)
        self.assertIn("fetch','--depth=1','--no-tags','origin',$Commit", installer)
        self.assertIn("verify_mv_adapter_env.py", installer)
        self.assertIn("mv-adapter-readiness.json", installer)
        self.assertIn("Test-PythonCommand -Python $TripoPython", installer)
        self.assertIn("/XF '*.pyc' 'local.json'", installer)


    def test_model_cache_partial_failure_is_retryable(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        prefetch = (ROOT / "scripts" / "prefetch_models.py").read_text(encoding="utf-8")
        self.assertIn("Test-JsonStatus $ModelReceipt 'passed'", installer)
        self.assertIn("--verify-only", installer)
        self.assertIn("local_files_only=args.verify_only", prefetch)
        self.assertIn('"status": "failed" if required_failed else "passed"', prefetch)
        self.assertIn("raise SystemExit(1)", prefetch)
        self.assertIn("--include-triposr", prefetch)

    def test_studio_native_runtime_is_explicitly_verified(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("require('express'); require('sqlite3')", installer)
        self.assertNotIn("Set-StudioScriptPolicy", installer)
        self.assertIn("npm rebuild sqlite3 --foreground-scripts --ignore-scripts=false", installer)
        self.assertIn("npm ci --no-audit --no-fund --ignore-scripts", installer)
        self.assertIn("npm rebuild sqlite3 --foreground-scripts --ignore-scripts=false", installer)

    def test_powershell_sources_have_balanced_braces(self):
        for relative in [
            "INSTALL-ONE-CLICK.ps1", "scripts/windows/install-checkpoints.ps1",
            "scripts/windows/SMOKE-SERVICES.ps1", "RESUME-LAST-JOB.ps1", "JOB-STATUS.ps1",
            "START-STUDIO.ps1", "STOP-STUDIO.ps1",
        ]:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertEqual(text.count("{"), text.count("}"), relative)
            self.assertEqual(text.count("("), text.count(")"), relative)


    def test_asset_jobs_are_resume_first(self):
        service = (ROOT / "service" / "app.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "src" / "lowvram3d" / "pipeline.py").read_text(encoding="utf-8")
        postprocess = (ROOT / "src" / "lowvram3d" / "postprocess.py").read_text(encoding="utf-8")
        registration = (ROOT / "scripts" / "register_studio.py").read_text(encoding="utf-8")
        self.assertIn('/v1/jobs/{job_id}/resume', service)
        self.assertIn('automatic_failed_card_resume', service)
        self.assertIn('failed_context_job', service)
        self.assertIn('resume_failed_job', registration)
        self.assertIn('full_contract', pipeline)
        self.assertIn('postprocess_contract', postprocess)
        self.assertTrue((ROOT / 'RESUME-LAST-JOB.cmd').is_file())
        self.assertIn('Resume Last 3D Job.lnk', (ROOT / 'INSTALL-ONE-CLICK.ps1').read_text(encoding='utf-8'))


    def test_git_checkouts_validate_commit_objects_and_required_files(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("function Test-GitCheckout", installer)
        self.assertIn('cat-file -e "$Commit^{commit}"', installer)
        self.assertIn(r"-RequiredFiles @('package.json','server.js','python-server\main.py')", installer)
        self.assertIn("remote','set-url','origin',$Url", installer)
        self.assertNotIn("if ((Get-GitHead $Path) -eq $Commit) { return }", installer)

    def test_runtime_start_is_health_checked_and_failure_clean(self):
        start = (ROOT / "START-STUDIO.ps1").read_text(encoding="utf-8")
        stop = (ROOT / "STOP-STUDIO.ps1").read_text(encoding="utf-8")
        self.assertIn("HealthUrl", start)
        self.assertIn("process_start_time", start)
        self.assertIn("Test-RecordMatchesProcess", start)
        self.assertIn("Startup failed; only processes started by this attempt were stopped.", start)
        self.assertIn("finally {", start)
        self.assertIn("SetEnvironmentVariable($key,$saved[$key],'Process')", start)
        self.assertIn("Test-RecordMatchesProcess", stop)
        self.assertIn("ConvertFrom-Json", stop)

    def test_avatar_pipeline_is_local_identity_and_dance_aware(self):
        pipeline = (ROOT / "src" / "lowvram3d" / "pipeline.py").read_text(encoding="utf-8")
        worker = (ROOT / "workers" / "avatar_preprocess.py").read_text(encoding="utf-8")
        rig = (ROOT / "blender" / "rig_animate.py").read_text(encoding="utf-8")
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("ZhengPeng7/BiRefNet", worker)
        self.assertIn("e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", worker)
        self.assertIn("mediapipe", worker)
        self.assertIn("decontaminate_edges", worker)
        self.assertIn("normalize_subject", worker)
        avatar = (ROOT / "src" / "lowvram3d" / "avatar.py").read_text(encoding="utf-8")
        self.assertIn("preprocess_subject", pipeline)
        self.assertIn("subject_preprocess", avatar)
        self.assertIn("Photorealistic full-body human avatar", pipeline)
        self.assertIn("dance_loop", rig)
        self.assertIn("pose_guided_proportions", rig)
        self.assertIn("mediapipe==0.10.21", installer)
        self.assertIn("models-sd21-mvadapter-birefnet-e2bf8e4", installer)
        self.assertIn("huggingface_hub==0.27.1", installer)


    def test_mv_adapter_readiness_is_diagnostic_and_uses_one_opencv_wheel(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        verifier = (ROOT / "scripts" / "verify_mv_adapter_env.py").read_text(encoding="utf-8")
        self.assertIn("opencv-contrib-python==4.11.0.86", installer)
        dependency_line = next(
            line for line in installer.splitlines()
            if "mediapipe==0.10.21" in line and "UvPip" in line
        )
        self.assertNotIn("'opencv-python'", dependency_line)
        self.assertIn("UvPip -Python $MvPython -Arguments (@('uninstall') + $OpenCvNames)", installer)
        self.assertIn("mv-adapter-readiness.log", installer)
        self.assertIn("opencv_single_distribution", verifier)
        self.assertIn("mediapipe_pose", verifier)
        self.assertIn("transformers_birefnet_api", verifier)
        self.assertIn("torch_cuda", verifier)
        self.assertIn("traceback", verifier)

    def test_control_environment_declares_and_verifies_cv2_and_httpx(self):
        # avatar_mask.py imports cv2 and starlette's TestClient imports httpx, so both are
        # hard control-environment dependencies. Stage 14 failed once because neither was
        # declared and stage 02's readiness probe did not look for them.
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        pinned = [
            line.strip()
            for line in (ROOT / "requirements-control.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertIn("opencv-python-headless==4.11.0.86", pinned)
        self.assertIn("httpx==0.27.2", pinned)
        opencv = [line for line in pinned if line.lower().startswith("opencv")]
        self.assertEqual(opencv, ["opencv-python-headless==4.11.0.86"], "exactly one opencv wheel")
        probe = next(
            line for line in installer.splitlines()
            if "Test-PythonCommand $ControlPython" in line
        )
        self.assertIn("cv2", probe)
        self.assertIn("httpx", probe)

    def test_package_verification_can_import_every_top_level_package(self):
        # Stage 14 discovered only 40 of 70 tests because PYTHONPATH carried src alone, so
        # 'import service', 'import workers' and 'import scripts' failed during discovery.
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        pythonpath = next(
            line for line in installer.splitlines()
            if "$env:PYTHONPATH" in line and "'src'" in line
        )
        self.assertIn("PathSeparator", pythonpath)
        self.assertIn("$AppRoot", pythonpath.split("PathSeparator", 1)[1])
        # -t would require tests/__init__.py, which is deliberately absent.
        self.assertFalse((ROOT / "tests" / "__init__.py").exists())
        discover = next(
            line for line in installer.splitlines() if "-m unittest discover" in line
        )
        self.assertNotIn("-t ", discover)

    def test_blender_stages_honour_the_inherited_pythonpath(self):
        # Blender's embedded interpreter ignores PYTHONPATH unless --python-use-system-env is
        # passed, so without it every blender/*.py script dies on "No module named 'common'".
        pipeline = (ROOT / "src" / "lowvram3d" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("--python-use-system-env", pipeline)
        self.assertIn('os.pathsep.join((str(self.package_root / "blender"), str(self.package_root / "src")))', pipeline)

    def test_lane_a_base_model_is_configurable(self):
        # stabilityai/stable-diffusion-2-1-base no longer resolves on Hugging Face, so the
        # MV-Adapter call sites must pass a configurable (local) base model.
        config = (ROOT / "src" / "lowvram3d" / "config.py").read_text(encoding="utf-8")
        self.assertIn("sd21_base_model", config)
        for module in ("pipeline.py", "appearance.py"):
            text = (ROOT / "src" / "lowvram3d" / module).read_text(encoding="utf-8")
            self.assertIn("--base-model", text, module)
            self.assertIn("sd21_base_model", text, module)

    def test_stage10_configuration_adds_optional_fields_safely(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        self.assertIn("function Set-ObjectProperty", installer)
        self.assertIn("Set-ObjectProperty -Object $cfg.extra -Name 'triposr_ready'", installer)
        self.assertIn("Set-ObjectProperty -Object $cfg.extra -Name 'triposr_backend'", installer)
        self.assertNotIn("$cfg.extra.triposr_ready =", installer)

    def test_avatar_alpha_is_preserved_and_animation_is_validated(self):
        views = (ROOT / "workers" / "make_fallback_views.py").read_text(encoding="utf-8")
        package = (ROOT / "blender" / "package_validate.py").read_text(encoding="utf-8")
        self.assertIn("preserved_source_alpha", views)
        self.assertIn('export_strategy == "animated_human_avatar"', package)
        self.assertIn("Animated avatar contains no dance action", package)
        self.assertIn('args.asset_type in {"avatar", "character", "creature"}', package)

    def test_birefnet_snapshot_is_pinned_for_prefetch_and_runtime(self):
        prefetch = (ROOT / "scripts" / "prefetch_models.py").read_text(encoding="utf-8")
        worker = (ROOT / "workers" / "avatar_preprocess.py").read_text(encoding="utf-8")
        revision = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
        self.assertIn(revision, prefetch)
        self.assertIn(revision, worker)
        self.assertIn("revision=revision", prefetch)
        self.assertIn("revision=revision", worker)

    def test_installer_runs_real_service_smoke(self):
        installer = (ROOT / "INSTALL-ONE-CLICK.ps1").read_text(encoding="utf-8")
        smoke = (ROOT / "scripts" / "windows" / "SMOKE-SERVICES.ps1").read_text(encoding="utf-8")
        self.assertIn("11-runtime-service-smoke", installer)
        self.assertIn("SMOKE-SERVICES.ps1", installer)
        self.assertIn("Test-JsonStatus $SmokeProof 'passed'", installer)
        self.assertIn("/health", smoke)
        self.assertIn("/api/settings", smoke)
        self.assertIn("taskkill.exe", smoke)
        self.assertIn("status = 'passed'", smoke)


if __name__ == "__main__":
    unittest.main()

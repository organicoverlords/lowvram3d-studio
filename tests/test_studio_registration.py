from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import register_studio


class StudioRegistrationTests(unittest.TestCase):
    def test_registers_asset_class_providers_and_graph(self):
        calls = []
        counter = {"node": 0}

        def fake(url, method="GET", body=None):
            calls.append((url, method, body))
            if url.endswith("/api/settings") and method == "GET":
                return {"apis": {"custom": [{"id": "keep", "name": "Existing"}]}}
            if url.endswith("/api/settings") and method == "POST":
                return body
            if url.endswith("/api/projects"):
                return {"id": 7, "name": body["name"]}
            if url.endswith("/api/graph/nodes"):
                counter["node"] += 1
                return {"id": counter["node"], **body}
            if url.endswith("/api/graph/connections"):
                return {"id": 99, **body}
            return {}

        with patch.object(register_studio, "wait", lambda *_: None), patch.object(register_studio, "request_json", side_effect=fake):
            result = register_studio.register("http://studio", "http://worker", "C:/ComfyUI", "http://127.0.0.1:8188", True)
        settings_post = next(body for url, method, body in calls if url.endswith("/api/settings") and method == "POST")
        providers = settings_post["apis"]["custom"]
        low = {p["id"]: p for p in providers if str(p.get("id", "")).startswith("lowvram3d_")}
        classes = ("avatar", "character", "creature", "vehicle", "prop", "building", "room", "scene", "level")
        expected = {"lowvram3d_full", "lowvram3d_generate", "lowvram3d_texture", "lowvram3d_rig"}
        expected.update({f"lowvram3d_full_{name}" for name in classes})
        expected.update({f"lowvram3d_post_{name}" for name in classes})
        self.assertEqual(set(low), expected)
        self.assertEqual(low["lowvram3d_full"]["type"], "mesh-generation")
        self.assertEqual(low["lowvram3d_full_creature"]["type"], "mesh-generation")
        self.assertEqual(low["lowvram3d_full_avatar"]["body"]["animation_preset"], "dance")
        self.assertTrue(low["lowvram3d_full_avatar"]["body"]["background_removal"])
        self.assertEqual(low["lowvram3d_full_creature"]["body"]["asset_type"], "creature")
        self.assertTrue(low["lowvram3d_full_creature"]["body"]["resume_failed_job"])
        self.assertEqual(low["lowvram3d_generate"]["type"], "mesh-generation")
        self.assertEqual(low["lowvram3d_texture"]["type"], "mesh-texturing")
        self.assertEqual(low["lowvram3d_rig"]["type"], "mesh-rigging")
        self.assertEqual(low["lowvram3d_post_character"]["type"], "mesh-texturing")
        self.assertEqual(low["lowvram3d_post_level"]["body"]["asset_type"], "level")
        self.assertTrue(low["lowvram3d_post_level"]["body"]["resume_failed_job"])
        self.assertEqual(result["project_id"], 7)
        self.assertEqual(len(result["node_ids"]), 4)


if __name__ == "__main__":
    unittest.main()

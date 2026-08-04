"""Auto-bootstrap the LowVRAM3D named-camera MCP toolset."""

import lowvram3d_scene_tools
import toolset_registry
from toolset_registry._registry_interface import register_toolset_classes

unreal = lowvram3d_scene_tools.unreal
registration_errors = register_toolset_classes([lowvram3d_scene_tools.LowVRAM3DSceneTools])
unreal.log("LOWVRAM3D_SCENE_TOOLS_LOADED")
if registration_errors:
    unreal.log_error("LOWVRAM3D_SCENE_TOOLS_REGISTRATION_ERRORS=" + str(registration_errors))
else:
    unreal.log("LOWVRAM3D_SCENE_TOOLS_REGISTERED")

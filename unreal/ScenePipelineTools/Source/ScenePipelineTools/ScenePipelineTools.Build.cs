using UnrealBuildTool;

public class ScenePipelineTools : ModuleRules
{
    public ScenePipelineTools(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "Json",
            "JsonUtilities",
            "ModelContextProtocol",
            "ToolsetRegistry",
            "UnrealEd"
        });
    }
}

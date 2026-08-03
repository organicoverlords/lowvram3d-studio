#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "ToolsetRegistry/ToolsetDefinition.h"

#include "ScenePipelineTools.generated.h"

class UToolCallAsyncResultString;

UCLASS(BlueprintType, Hidden)
class UScenePipelineTools final : public UToolsetDefinition
{
    GENERATED_BODY()

public:
    UFUNCTION(meta = (AICallable), Category = "Scene Pipeline")
    static FString scene_pipeline_start_job(const FString& workflow_name, const FString& idempotency_key,
        const FString& expected_project, const FString& expected_map);

    UFUNCTION(meta = (AICallable), Category = "Scene Pipeline")
    static FString scene_pipeline_get_job_status(const FString& job_id);

    UFUNCTION(meta = (AICallable), Category = "Scene Pipeline")
    static FString scene_pipeline_cancel_job(const FString& job_id);
};

class FScenePipelineToolsModule final : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    static FScenePipelineToolsModule& Get();

    FString StartJob(const FString& WorkflowName, const FString& IdempotencyKey,
        const FString& ExpectedProject, const FString& ExpectedMap);
    FString GetJobStatus(const FString& JobId) const;
    FString CancelJob(const FString& JobId);

private:
    bool Tick(float DeltaSeconds);
    void RegisterToolset();
    void UnregisterToolset();
    void FailJob(const FString& Error);
    FString MakeResult(bool bOk, const FString& Message, const TSharedPtr<class FJsonObject>& Details = {}) const;
    bool WriteJsonAtomic(const FString& FilePath, const TSharedPtr<class FJsonObject>& Object) const;
    bool WriteStepReceipt(const FString& State, const FString& Message);
    bool IsProtectedMap(const FString& MapPath) const;
    FString CurrentMapPath() const;
    bool VerifyCube(bool& bOutFound) const;

    struct FJob
    {
        FString JobId;
        FString WorkflowName;
        FString IdempotencyKey;
        FString ExpectedProject;
        FString ExpectedMap;
        FString ReceiptDir;
        int32 Phase = 0;
        FString State = TEXT("ACCEPTED");
        FString Error;
        FString CubeName = TEXT("AgentProof_DiagnosticCube");
        bool bCancelled = false;
    };

    TUniquePtr<FJob> ActiveJob;
    FTSTicker::FDelegateHandle TickerHandle;
    FDelegateHandle PostEngineInitHandle;
    bool bToolsetRegistered = false;
    uint32 StepCounter = 0;
};

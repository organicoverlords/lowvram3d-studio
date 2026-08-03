#include "ScenePipelineTools.h"

#include "AssetSelection.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Editor.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "EngineUtils.h"
#include "FileHelpers.h"
#include "HAL/FileManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "ToolsetRegistry/UToolsetRegistry.h"

namespace
{
    FString JsonString(const TSharedPtr<FJsonObject>& Object)
    {
        FString Out;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
        FJsonSerializer::Serialize(Object.ToSharedRef(), Writer);
        return Out;
    }

    TSharedPtr<FJsonObject> NewObject()
    {
        return MakeShared<FJsonObject>();
    }
}

IMPLEMENT_MODULE(FScenePipelineToolsModule, ScenePipelineTools)

FString UScenePipelineTools::scene_pipeline_start_job(const FString& workflow_name, const FString& idempotency_key,
    const FString& expected_project, const FString& expected_map)
{
    return FScenePipelineToolsModule::Get().StartJob(workflow_name, idempotency_key, expected_project, expected_map);
}

FString UScenePipelineTools::scene_pipeline_get_job_status(const FString& job_id)
{
    return FScenePipelineToolsModule::Get().GetJobStatus(job_id);
}

FString UScenePipelineTools::scene_pipeline_cancel_job(const FString& job_id)
{
    return FScenePipelineToolsModule::Get().CancelJob(job_id);
}

FScenePipelineToolsModule& FScenePipelineToolsModule::Get()
{
    return FModuleManager::LoadModuleChecked<FScenePipelineToolsModule>(TEXT("ScenePipelineTools"));
}

void FScenePipelineToolsModule::StartupModule()
{
    TickerHandle = FTSTicker::GetCoreTicker().AddTicker(
        FTickerDelegate::CreateRaw(this, &FScenePipelineToolsModule::Tick), 0.0f);
    PostEngineInitHandle = FCoreDelegates::OnPostEngineInit.AddRaw(this, &FScenePipelineToolsModule::RegisterToolset);
    if (GEditor)
    {
        RegisterToolset();
    }
}

void FScenePipelineToolsModule::ShutdownModule()
{
    if (TickerHandle.IsValid())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
        TickerHandle.Reset();
    }
    if (PostEngineInitHandle.IsValid())
    {
        FCoreDelegates::OnPostEngineInit.Remove(PostEngineInitHandle);
        PostEngineInitHandle.Reset();
    }
    UnregisterToolset();
    ActiveJob.Reset();
}

void FScenePipelineToolsModule::RegisterToolset()
{
    if (!bToolsetRegistered && UToolsetRegistry::IsAvailable())
    {
        UToolsetRegistry::RegisterToolsetClass(UScenePipelineTools::StaticClass());
        bToolsetRegistered = UToolsetRegistry::IsToolsetClassRegistered(UScenePipelineTools::StaticClass());
    }
}

void FScenePipelineToolsModule::UnregisterToolset()
{
    if (bToolsetRegistered && UToolsetRegistry::IsAvailable())
    {
        UToolsetRegistry::UnregisterToolsetClass(UScenePipelineTools::StaticClass());
    }
    bToolsetRegistered = false;
}

FString FScenePipelineToolsModule::MakeResult(bool bOk, const FString& Message,
    const TSharedPtr<FJsonObject>& Details) const
{
    TSharedPtr<FJsonObject> Result = NewObject();
    Result->SetBoolField(TEXT("ok"), bOk);
    Result->SetStringField(TEXT("message"), Message);
    if (Details.IsValid())
    {
        Result->SetObjectField(TEXT("details"), Details);
    }
    return JsonString(Result);
}

bool FScenePipelineToolsModule::WriteJsonAtomic(const FString& FilePath, const TSharedPtr<FJsonObject>& Object) const
{
    const FString TempPath = FilePath + TEXT(".tmp");
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(FilePath), true);
    if (!FFileHelper::SaveStringToFile(JsonString(Object), *TempPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
    {
        return false;
    }
    IFileManager::Get().Delete(*FilePath, false, true, true);
    return IFileManager::Get().Move(*FilePath, *TempPath, true, true, false, true);
}

bool FScenePipelineToolsModule::WriteStepReceipt(const FString& State, const FString& Message)
{
    if (!ActiveJob.IsValid())
    {
        return false;
    }
    ActiveJob->State = State;
    TSharedPtr<FJsonObject> Receipt = NewObject();
    Receipt->SetStringField(TEXT("job_id"), ActiveJob->JobId);
    Receipt->SetStringField(TEXT("workflow"), ActiveJob->WorkflowName);
    Receipt->SetNumberField(TEXT("step"), static_cast<double>(++StepCounter));
    Receipt->SetStringField(TEXT("state"), State);
    Receipt->SetStringField(TEXT("message"), Message);
    Receipt->SetStringField(TEXT("timestamp_utc"), FDateTime::UtcNow().ToIso8601());
    const FString Name = FString::Printf(TEXT("step-%04u-%s.json"), StepCounter, *State);
    return WriteJsonAtomic(FPaths::Combine(ActiveJob->ReceiptDir, Name), Receipt);
}

FString FScenePipelineToolsModule::CurrentMapPath() const
{
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        return FString();
    }
    return GEditor->GetEditorWorldContext().World()->GetOutermost()->GetName();
}

bool FScenePipelineToolsModule::IsProtectedMap(const FString& MapPath) const
{
    return MapPath.Contains(TEXT("Castlegrounds"), ESearchCase::IgnoreCase)
        || MapPath.Contains(TEXT("L_Castlegrounds"), ESearchCase::IgnoreCase)
        || MapPath.Contains(TEXT("Source"), ESearchCase::IgnoreCase);
}

FString FScenePipelineToolsModule::StartJob(const FString& WorkflowName, const FString& IdempotencyKey,
    const FString& ExpectedProject, const FString& ExpectedMap)
{
    if (WorkflowName != TEXT("diagnostic_cube_v1"))
    {
        return MakeResult(false, TEXT("UNKNOWN_WORKFLOW"));
    }
    if (ExpectedMap != TEXT("/Game/AgentProof/MCP/L_MCP_Diagnostic"))
    {
        return MakeResult(false, TEXT("SOURCE_MAP_PROTECTION_REJECTED"));
    }
    if (ExpectedProject.IsEmpty() || IdempotencyKey.IsEmpty())
    {
        return MakeResult(false, TEXT("MISSING_REQUIRED_JOB_FIELDS"));
    }
    if (ActiveJob.IsValid())
    {
        if (ActiveJob->IdempotencyKey == IdempotencyKey)
        {
            TSharedPtr<FJsonObject> Details = NewObject();
            Details->SetStringField(TEXT("job_id"), ActiveJob->JobId);
            Details->SetStringField(TEXT("state"), ActiveJob->State);
            return MakeResult(true, TEXT("IDEMPOTENT_EXISTING_JOB"), Details);
        }
        return MakeResult(false, TEXT("ANOTHER_JOB_ACTIVE"));
    }

    ActiveJob = MakeUnique<FJob>();
    ActiveJob->JobId = FString::Printf(TEXT("diagnostic-%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
    ActiveJob->WorkflowName = WorkflowName;
    ActiveJob->IdempotencyKey = IdempotencyKey;
    ActiveJob->ExpectedProject = ExpectedProject;
    ActiveJob->ExpectedMap = ExpectedMap;
    ActiveJob->ReceiptDir = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("AgentMCPJobs"), ActiveJob->JobId);
    StepCounter = 0;

    TSharedPtr<FJsonObject> Accepted = NewObject();
    Accepted->SetStringField(TEXT("job_id"), ActiveJob->JobId);
    Accepted->SetStringField(TEXT("workflow"), WorkflowName);
    Accepted->SetStringField(TEXT("idempotency_key"), IdempotencyKey);
    Accepted->SetStringField(TEXT("expected_project"), ExpectedProject);
    Accepted->SetStringField(TEXT("expected_map"), ExpectedMap);
    Accepted->SetStringField(TEXT("retry_policy"), TEXT("never_auto_retry_mutation"));
    Accepted->SetStringField(TEXT("state"), TEXT("ACCEPTED"));
    if (!WriteJsonAtomic(FPaths::Combine(ActiveJob->ReceiptDir, TEXT("accepted.json")), Accepted))
    {
        ActiveJob.Reset();
        return MakeResult(false, TEXT("ACCEPTANCE_RECEIPT_FAILED"));
    }

    TSharedPtr<FJsonObject> Preflight = NewObject();
    Preflight->SetStringField(TEXT("job_id"), ActiveJob->JobId);
    Preflight->SetStringField(TEXT("state"), TEXT("PREFLIGHT"));
    Preflight->SetStringField(TEXT("current_map"), CurrentMapPath());
    Preflight->SetBoolField(TEXT("source_map_protection"), !IsProtectedMap(CurrentMapPath()));
    Preflight->SetStringField(TEXT("execution_policy"), TEXT("one_operation_per_editor_tick"));
    if (IsProtectedMap(CurrentMapPath()) || !WriteJsonAtomic(FPaths::Combine(ActiveJob->ReceiptDir, TEXT("preflight.json")), Preflight))
    {
        FailJob(TEXT("PREFLIGHT_REJECTED"));
        return MakeResult(false, TEXT("PREFLIGHT_REJECTED"));
    }
    ActiveJob->State = TEXT("PREFLIGHT");
    return MakeResult(true, TEXT("ACCEPTED"), Accepted);
}

FString FScenePipelineToolsModule::GetJobStatus(const FString& JobId) const
{
    if (!ActiveJob.IsValid() || ActiveJob->JobId != JobId)
    {
        return MakeResult(false, TEXT("JOB_NOT_FOUND"));
    }
    TSharedPtr<FJsonObject> Details = NewObject();
    Details->SetStringField(TEXT("job_id"), ActiveJob->JobId);
    Details->SetStringField(TEXT("workflow"), ActiveJob->WorkflowName);
    Details->SetStringField(TEXT("state"), ActiveJob->State);
    Details->SetStringField(TEXT("current_map"), CurrentMapPath());
    Details->SetStringField(TEXT("receipt_dir"), ActiveJob->ReceiptDir);
    if (!ActiveJob->Error.IsEmpty())
    {
        Details->SetStringField(TEXT("error"), ActiveJob->Error);
    }
    return MakeResult(true, TEXT("STATUS"), Details);
}

FString FScenePipelineToolsModule::CancelJob(const FString& JobId)
{
    if (!ActiveJob.IsValid() || ActiveJob->JobId != JobId)
    {
        return MakeResult(false, TEXT("JOB_NOT_FOUND"));
    }
    ActiveJob->bCancelled = true;
    return MakeResult(true, TEXT("CANCEL_REQUESTED"));
}

void FScenePipelineToolsModule::FailJob(const FString& Error)
{
    if (!ActiveJob.IsValid())
    {
        return;
    }
    ActiveJob->Error = Error;
    WriteStepReceipt(TEXT("FAILED"), Error);
    TSharedPtr<FJsonObject> Final = NewObject();
    Final->SetStringField(TEXT("job_id"), ActiveJob->JobId);
    Final->SetStringField(TEXT("state"), TEXT("FAILED"));
    Final->SetStringField(TEXT("error"), Error);
    WriteJsonAtomic(FPaths::Combine(ActiveJob->ReceiptDir, TEXT("final.json")), Final);
}

bool FScenePipelineToolsModule::VerifyCube(bool& bOutFound) const
{
    bOutFound = false;
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        return false;
    }
    for (TActorIterator<AStaticMeshActor> It(GEditor->GetEditorWorldContext().World()); It; ++It)
    {
        if (It->GetActorLabel() == ActiveJob->CubeName && It->GetStaticMeshComponent()
            && It->GetStaticMeshComponent()->GetStaticMesh()
            && It->GetStaticMeshComponent()->GetStaticMesh()->GetPathName() == TEXT("/Engine/BasicShapes/Cube.Cube"))
        {
            bOutFound = true;
            return true;
        }
    }
    return true;
}

bool FScenePipelineToolsModule::Tick(float)
{
    if (!ActiveJob.IsValid() || ActiveJob->State == TEXT("COMPLETED") || ActiveJob->State == TEXT("FAILED"))
    {
        return true;
    }
    if (ActiveJob->bCancelled)
    {
        FailJob(TEXT("CANCELLED"));
        return true;
    }
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    const FString MapFilename = FPackageName::LongPackageNameToFilename(
        ActiveJob->ExpectedMap, FPackageName::GetMapPackageExtension());

    switch (ActiveJob->Phase++)
    {
    case 0:
        if (!GEditor || IsProtectedMap(CurrentMapPath())) { FailJob(TEXT("PREFLIGHT_REJECTED")); break; }
        GEditor->CreateNewMapForEditing(false, false);
        WriteStepReceipt(TEXT("MAP_CREATED"), TEXT("Created empty editor map"));
        break;
    case 1:
        if (!GEditor || !GEditor->GetEditorWorldContext().World()) { FailJob(TEXT("MAP_NOT_READY")); break; }
        WriteStepReceipt(TEXT("MAP_READY"), TEXT("Empty diagnostic world is ready"));
        break;
    case 2:
        WriteStepReceipt(TEXT("SAVE_REQUESTED"), TEXT("Initial save requested"));
        break;
    case 3:
        if (!World || !FEditorFileUtils::SaveMap(World, MapFilename)) { FailJob(TEXT("INITIAL_SAVE_FAILED")); break; }
        WriteStepReceipt(TEXT("SAVED"), TEXT("Initial diagnostic map saved"));
        break;
    case 4:
        WriteStepReceipt(TEXT("RELOAD_REQUESTED"), TEXT("Initial reload requested"));
        break;
    case 5:
        if (!FEditorFileUtils::LoadMap(MapFilename, false, false)) { FailJob(TEXT("INITIAL_RELOAD_FAILED")); break; }
        WriteStepReceipt(TEXT("RELOADED"), TEXT("Initial diagnostic map reloaded"));
        break;
    case 6:
        World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
        if (!World) { FailJob(TEXT("WORLD_MISSING_BEFORE_CUBE")); break; }
        {
            UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
            AStaticMeshActor* Cube = CubeMesh ? World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), FTransform(FRotator::ZeroRotator, FVector(100, 0, 50))) : nullptr;
            if (!Cube) { FailJob(TEXT("CUBE_SPAWN_FAILED")); break; }
            Cube->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
            Cube->SetActorLabel(ActiveJob->CubeName);
            Cube->SetActorTransform(FTransform(FRotator::ZeroRotator, FVector(100, 0, 50), FVector(1, 1, 1)));
            Cube->MarkPackageDirty();
        }
        WriteStepReceipt(TEXT("CUBE_SPAWNED"), TEXT("Deterministic engine cube spawned"));
        break;
    case 7:
        WriteStepReceipt(TEXT("SAVE_REQUESTED"), TEXT("Cube save requested"));
        break;
    case 8:
        if (!World || !FEditorFileUtils::SaveMap(World, MapFilename)) { FailJob(TEXT("CUBE_SAVE_FAILED")); break; }
        WriteStepReceipt(TEXT("SAVED"), TEXT("Cube-containing map saved"));
        break;
    case 9:
        WriteStepReceipt(TEXT("RELOAD_REQUESTED"), TEXT("Cube reload requested"));
        break;
    case 10:
        if (!FEditorFileUtils::LoadMap(MapFilename, false, false)) { FailJob(TEXT("CUBE_RELOAD_FAILED")); break; }
        WriteStepReceipt(TEXT("RELOADED"), TEXT("Cube-containing map reloaded"));
        break;
    case 11:
        {
            bool bFound = false;
            if (!VerifyCube(bFound) || !bFound) { FailJob(TEXT("CUBE_PERSISTENCE_FAILED")); break; }
            WriteStepReceipt(TEXT("VALIDATED"), TEXT("Cube persisted after save and reload"));
            ActiveJob->State = TEXT("COMPLETED");
            TSharedPtr<FJsonObject> Final = NewObject();
            Final->SetStringField(TEXT("job_id"), ActiveJob->JobId);
            Final->SetStringField(TEXT("workflow"), ActiveJob->WorkflowName);
            Final->SetStringField(TEXT("state"), TEXT("COMPLETED"));
            Final->SetBoolField(TEXT("cube_persisted"), true);
            Final->SetStringField(TEXT("map"), ActiveJob->ExpectedMap);
            Final->SetStringField(TEXT("timestamp_utc"), FDateTime::UtcNow().ToIso8601());
            WriteJsonAtomic(FPaths::Combine(ActiveJob->ReceiptDir, TEXT("final.json")), Final);
        }
        break;
    default:
        FailJob(TEXT("INVALID_STATE"));
        break;
    }
    return true;
}

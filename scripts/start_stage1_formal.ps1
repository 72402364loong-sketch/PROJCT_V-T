param(
    [string]$PythonExe = 'C:\Users\shaw1\AppData\Local\Programs\Python\Python310\python.exe',
    [string]$StageConfig = 'configs/stages/stage1_perception_fold1.yaml',
    [switch]$ForceRestart,
    [switch]$StartFresh,
    [ValidateSet('Hidden', 'Minimized', 'Normal')]
    [string]$WindowStyle = 'Minimized'
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$runDir = Join-Path $projectRoot 'runs\stage1_perception_fold1'
$tmpDir = Join-Path $projectRoot '.codex_tmp'
$hostName = 'CMG_Stage1_Perception_Fold1'
$runnerPath = Join-Path $tmpDir 'stage1_formal_host_runner.ps1'
$stdoutPath = Join-Path $runDir 'hosted_stage1.stdout.log'
$stderrPath = Join-Path $runDir 'hosted_stage1.stderr.log'
$launchInfoPath = Join-Path $runDir 'hosted_stage1.launch.json'
$resumeCheckpoint = Join-Path $runDir 'checkpoints\stage1_latest.pt'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$existingPid = $null
if (Test-Path $launchInfoPath) {
    try {
        $existingLaunch = Get-Content -Raw -Encoding UTF8 $launchInfoPath | ConvertFrom-Json
        if ($existingLaunch.pid) {
            $existingPid = [int]$existingLaunch.pid
        }
    } catch {
        $existingPid = $null
    }
}

if ($null -ne $existingPid) {
    $existingProc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($null -ne $existingProc) {
        if (-not $ForceRestart) {
            throw "Host process $existingPid is already running. Use scripts/watch_stage1_formal.ps1 to inspect it, or rerun with -ForceRestart."
        }
        Stop-Process -Id $existingPid -Force
        Start-Sleep -Seconds 1
    }
}

if (Test-Path $stdoutPath) {
    Remove-Item -Force $stdoutPath
}
if (Test-Path $stderrPath) {
    Remove-Item -Force $stderrPath
}

$trainCommand = "& '$PythonExe' scripts/train.py --project-root . --stage '$StageConfig'"
$mode = 'fresh'
$resumePathForLog = $null
if ((-not $StartFresh) -and (Test-Path $resumeCheckpoint)) {
    $trainCommand += " --resume '$resumeCheckpoint'"
    $mode = 'resume_latest'
    $resumePathForLog = $resumeCheckpoint
}
$trainCommand += " 1>>'$stdoutPath' 2>>'$stderrPath'"

$runnerBody = @"
`$ErrorActionPreference = 'Stop'
`$env:PYTHONUNBUFFERED = '1'
Set-Location '$projectRoot'
$trainCommand
"@
Set-Content -Path $runnerPath -Value $runnerBody -Encoding UTF8

$startParams = @{
    FilePath = 'C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe'
    ArgumentList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runnerPath)
    WorkingDirectory = $projectRoot
    PassThru = $true
}
if ($WindowStyle -ne 'Hidden') {
    $startParams['WindowStyle'] = $WindowStyle
}
$proc = Start-Process @startParams
Start-Sleep -Seconds 2
$proc.Refresh()

$launchInfo = [ordered]@{
    host_name = $hostName
    pid = $proc.Id
    has_exited = $proc.HasExited
    started_at = (Get-Date).ToString('s')
    mode = $mode
    resume_checkpoint = $resumePathForLog
    run_dir = $runDir
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    runner_script = $runnerPath
    python_exe = $PythonExe
    stage_config = $StageConfig
    window_style = $WindowStyle
}
$launchInfo | ConvertTo-Json -Depth 4 | Set-Content -Path $launchInfoPath -Encoding UTF8

Write-Output "Stage1 host process started."
Write-Output "HostName: $hostName"
Write-Output "PID: $($proc.Id)"
Write-Output "WindowStyle: $WindowStyle"
Write-Output "Mode: $mode"
if ($null -ne $resumePathForLog) {
    Write-Output "ResumeCheckpoint: $resumePathForLog"
}
Write-Output "RunDir: $runDir"
Write-Output "StdoutLog: $stdoutPath"
Write-Output "StderrLog: $stderrPath"
Write-Output "WatchCommand: powershell -ExecutionPolicy Bypass -File scripts/watch_stage1_formal.ps1 -Follow"

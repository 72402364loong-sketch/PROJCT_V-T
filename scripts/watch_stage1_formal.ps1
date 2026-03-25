param(
    [switch]$Follow,
    [int]$Tail = 40
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$runDir = Join-Path $projectRoot 'runs\stage1_perception_fold1'
$launchInfoPath = Join-Path $runDir 'hosted_stage1.launch.json'
$stdoutPath = Join-Path $runDir 'hosted_stage1.stdout.log'
$stderrPath = Join-Path $runDir 'hosted_stage1.stderr.log'
$pidValue = $null
$hostName = 'CMG_Stage1_Perception_Fold1'

if (Test-Path $launchInfoPath) {
    $launchInfo = Get-Content -Raw -Encoding UTF8 $launchInfoPath | ConvertFrom-Json
    $hostName = [string]$launchInfo.host_name
    $stdoutPath = [string]$launchInfo.stdout_log
    $stderrPath = [string]$launchInfo.stderr_log
    if ($launchInfo.pid) {
        $pidValue = [int]$launchInfo.pid
    }
}

Write-Output "RunDir: $runDir"
Write-Output "HostName: $hostName"
if ($null -ne $pidValue) {
    Write-Output "PID: $pidValue"
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-Output 'ProcessState: running'
    } else {
        Write-Output 'ProcessState: exited'
    }
} else {
    Write-Output 'PID: unknown'
}

Write-Output ''
Write-Output 'Recent metrics.csv:'
if (Test-Path (Join-Path $runDir 'metrics.csv')) {
    Get-Content -Path (Join-Path $runDir 'metrics.csv') -Tail 5 -Encoding UTF8
} else {
    Write-Output 'metrics.csv not created yet.'
}

Write-Output ''
Write-Output "Recent stderr log ($stderrPath):"
if (Test-Path $stderrPath) {
    if ($Follow) {
        Get-Content -Path $stderrPath -Tail $Tail -Wait -Encoding UTF8
    } else {
        Get-Content -Path $stderrPath -Tail $Tail -Encoding UTF8
    }
} else {
    Write-Output 'stderr log not created yet.'
}

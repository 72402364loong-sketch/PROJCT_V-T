param()

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$runDir = Join-Path $projectRoot 'runs\stage1_perception_fold1'
$launchInfoPath = Join-Path $runDir 'hosted_stage1.launch.json'

if (-not (Test-Path $launchInfoPath)) {
    Write-Output 'No hosted Stage1 launch record found.'
    exit 0
}

$launchInfo = Get-Content -Raw -Encoding UTF8 $launchInfoPath | ConvertFrom-Json
if (-not $launchInfo.pid) {
    Write-Output 'Launch record does not contain a PID.'
    exit 0
}

$pidValue = [int]$launchInfo.pid
$proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
if ($null -eq $proc) {
    Write-Output "Process $pidValue is already stopped."
    exit 0
}

Stop-Process -Id $pidValue -Force
Write-Output "Process $pidValue stopped."

# VoxEMW Windows start/stop (pipeline :8765 + orchestrator :8000)
# GPT-SoVITS (:8899) stays in WSL; this script only manages the two Windows services
# that read the repo directly - no more syncing code changes to the WSL copy.
#
# Usage (PowerShell):
#   powershell -File scripts/start_assistant_win.ps1            # start
#   powershell -File scripts/start_assistant_win.ps1 stop       # stop
#   powershell -File scripts/start_assistant_win.ps1 status     # status
#
# Note: stop WSL pipeline/orchestrator first (port conflict). ASCII-only file
# on purpose - PS 5.1 misreads UTF-8-without-BOM as ANSI.
# The venv is uv-managed: .venv\Scripts\python.exe is a launcher shim that spawns
# the real interpreter (base python), so EACH service shows up as two processes.
# All matching PIDs are tracked/killed together on stop.
param([string]$Action = "start")

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Config = "configs\assistant.yaml"
$Logs = Join-Path $Repo "logs"
if (-not (Test-Path $Logs)) { New-Item -ItemType Directory -Force $Logs | Out-Null }

# Same env as the WSL script: offline HF cache, bounded threads.
$env:HF_HUB_OFFLINE = "1"
$env:OMP_NUM_THREADS = "4"
$env:OMP_WAIT_POLICY = "PASSIVE"
$env:MKL_NUM_THREADS = "4"

function Test-Port([int]$port) {
    (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -ne $null
}

# All PIDs whose cmdline matches the module name (launcher shim + real interpreter).
function Get-SvcPids([string]$name) {
    $results = @()
    $pidFile = Join-Path $Logs "$name.pid"
    if (Test-Path $pidFile) {
        $val = (Get-Content $pidFile -ErrorAction SilentlyContinue).Trim()
        if ($val -and (Get-Process -Id $val -ErrorAction SilentlyContinue)) { $results += [int]$val }
    }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$name*" } |
        ForEach-Object { $results += [int]$_.ProcessId }
    @($results | Sort-Object -Unique)
}

function Start-Svc([string]$name, [string]$module) {
    if ((Get-SvcPids $name).Count -gt 0) { Write-Host "$name already running"; return }
    Write-Host "Starting $name ($module)..."
    $p = Start-Process -FilePath $Py -ArgumentList "-m", $module, "--config", $Config `
        -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $Logs "$name.out.log") `
        -RedirectStandardError (Join-Path $Logs "$name.err.log")
    $p.Id | Out-File (Join-Path $Logs "$name.pid") -Encoding ascii
    Write-Host "    launcher PID=$($p.Id), log logs/$name.{out,err}.log"
}

function Wait-Port([int]$port, [int]$timeoutSec, [string]$label) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $timeoutSec) {
        if (Test-Port $port) { Write-Host "    $label ready ($([int]$sw.Elapsed.TotalSeconds)s)"; return $true }
        Start-Sleep -Seconds 2
    }
    Write-Host "    WARNING: $label not ready after ${timeoutSec}s"
    return $false
}

function Write-Status {
    foreach ($svc in @(@("pipeline", "voxemw.pipeline.launch", 8765), @("orchestrator", "voxemw.avatar.orchestrator", 8000))) {
        $name = $svc[0]; $mod = $svc[1]; $port = $svc[2]
        $pids = Get-SvcPids $mod
        $state = if ($pids.Count -gt 0) { "RUNNING (PID $($pids -join ','))" } else { "stopped" }
        $portUp = if (Test-Port $port) { " :$port LISTEN" } else { "" }
        Write-Host ("{0,-12}: {1}{2}" -f $name, $state, $portUp)
    }
    $gpu = if (Test-Port 8899) { "RUNNING (:8899)" } else { "stopped (start it in WSL)" }
    Write-Host ("{0,-12}: {1}" -f "gptsovits", $gpu)
}

switch ($Action) {
    "stop" {
        foreach ($name in @("pipeline", "orchestrator")) {
            $pids = Get-SvcPids $name
            foreach ($procId in $pids) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Host "Stopped $name (PID $procId)"
            }
        }
        Start-Sleep -Seconds 3
        Write-Status
    }
    "status" { Write-Status }
    default {
        if (-not (Test-Port 8899)) {
            Write-Warning "GPT-SoVITS (:8899) is DOWN - start the WSL services first"
        }
        Start-Svc "pipeline" "voxemw.pipeline.launch"
        Wait-Port 8765 60 "pipeline(:8765)"
        Start-Svc "orchestrator" "voxemw.avatar.orchestrator"
        Wait-Port 8000 30 "orchestrator(:8000)"
        Write-Host ""
        Write-Host "Open http://localhost:8000"
    }
}

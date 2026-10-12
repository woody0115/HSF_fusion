# Unattended chain for the event-only baseline (runs in its own window, survives this session):
#   1. wait until the DSEC download finishes ("ALL DONE" in D:\DSEC\download.log)
#   2. preprocess the remaining sequences (GEAR for every sequence)
#   3. rebuild the YOLO view
#   4. wait until no other training is using the GPU, then train the event-only YOLOv8s
# Progress: VS Code task "觀察訓練過程" (training) / "觀察下載進度" (download); own log below.
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$log = Join-Path $root 'runs\auto_event_baseline.log'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force (Join-Path $root 'runs') | Out-Null
function Log($m) { $l = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $l; Add-Content $log $l -Encoding utf8 }

Log 'waiting for the DSEC download to finish ...'
while (-not ((Test-Path D:\DSEC\download.log) -and ((Get-Content D:\DSEC\download.log -Tail 1) -match 'ALL DONE'))) {
    if ((Get-Content D:\DSEC\download.log -Tail 1) -match 'FAILED') { Log 'download reported FAILED; stopping (restart the download, then this script)'; exit 1 }
    Start-Sleep 60
}
Log 'download finished -> preprocessing'
& $py (Join-Path $root 'tools\preprocess_dsec.py') --workers 6 2>&1 | ForEach-Object { Log "  $_" }
& $py (Join-Path $root 'tools\build_yolo_view.py') 2>&1 | ForEach-Object { Log "  $_" }

Log 'waiting for the GPU (another training run may still be active) ...'
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_single\.py' }) { Start-Sleep 60 }
Log 'starting event-only YOLOv8s baseline'
& $py (Join-Path $root 'tools\train_single.py') --modality event
Log "event baseline finished (exit $LASTEXITCODE)"

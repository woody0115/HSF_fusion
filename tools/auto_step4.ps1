# Unattended: start step 4 (dual-stream "add" baseline, 20 epochs) as soon as the event-only
# baseline of step 3 has finished (its weights initialise the event backbone).
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$chainLog = Join-Path $root 'runs\auto_event_baseline.log'
$evtWeights = Join-Path $root 'runs\step3\event_yolov8s\weights\best.pt'
$log = Join-Path $root 'runs\auto_step4.log'
$env:PYTHONIOENCODING = 'utf-8'
function Log($m) { $l = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $l; Add-Content $log $l -Encoding utf8 }

Log 'waiting for the event-only baseline (step 3) to finish ...'
while (-not ((Test-Path $chainLog) -and (Select-String -Path $chainLog -Pattern 'event baseline finished' -Quiet))) { Start-Sleep 60 }
if (-not (Test-Path $evtWeights)) { Log "event weights missing ($evtWeights); not starting step 4"; exit 1 }
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_single\.py|train_dual\.py' }) { Start-Sleep 60 }
Log 'starting step 4: dual-stream add baseline, 20 epochs'
& $py (Join-Path $root 'tools\train_dual.py') --fusion add --epochs 20 --name dual_add_e20
Log "step 4 finished (exit $LASTEXITCODE)"

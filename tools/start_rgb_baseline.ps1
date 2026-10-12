# Unattended chain for the RGB-only baseline (own window, survives this session):
#   1. label stage for every sequence (copies RGB frames from the HDD to the SSD)
#   2. rebuild the YOLO view
#   3. train the RGB-only YOLOv8s (watch with the VS Code task "觀察訓練過程")
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$log = Join-Path $root 'runs\start_rgb_baseline.log'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force (Join-Path $root 'runs') | Out-Null
function Log($m) { $l = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $l; Add-Content $log $l -Encoding utf8 }

Log 'label stage + RGB copy to SSD ...'
& $py (Join-Path $root 'tools\preprocess_dsec.py') --workers 6 --labels-only 2>&1 | ForEach-Object { Log "  $_" }
& $py (Join-Path $root 'tools\build_yolo_view.py') 2>&1 | ForEach-Object { Log "  $_" }
Log 'starting RGB-only YOLOv8s baseline'
& $py (Join-Path $root 'tools\train_single.py') --modality rgb
Log "RGB baseline finished (exit $LASTEXITCODE)"

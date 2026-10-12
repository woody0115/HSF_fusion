# Unattended: start step 5 (dual-stream + AF, 20 epochs) after step 4 has finished.
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$prevLog = Join-Path $root 'runs\auto_step4.log'
$log = Join-Path $root 'runs\auto_step5.log'
$env:PYTHONIOENCODING = 'utf-8'
function Log($m) { $l = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $l; Add-Content $log $l -Encoding utf8 }

Log 'waiting for step 4 to finish ...'
while (-not ((Test-Path $prevLog) -and (Select-String -Path $prevLog -Pattern 'step 4 finished' -Quiet))) { Start-Sleep 60 }
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_single\.py|train_dual\.py' }) { Start-Sleep 60 }
Log 'starting step 5: dual-stream + AF, 20 epochs'
& $py (Join-Path $root 'tools\train_dual.py') --fusion af --epochs 20 --name dual_af_e20
Log "step 5 finished (exit $LASTEXITCODE)"

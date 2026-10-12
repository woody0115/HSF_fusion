# Unattended: run step 7 (DSEC-OWOD Task 1-4) after step 6 has finished.
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$prevLog = Join-Path $root 'runs\auto_step6.log'
$log = Join-Path $root 'runs\auto_step7.log'
$env:PYTHONIOENCODING = 'utf-8'
function Log($m) { $l = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $l; Add-Content $log $l -Encoding utf8 }

Log 'waiting for step 6 to finish ...'
while (-not ((Test-Path $prevLog) -and (Select-String -Path $prevLog -Pattern 'step 6 finished' -Quiet))) { Start-Sleep 60 }
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_single\.py|train_dual\.py|train_owod\.py' }) { Start-Sleep 60 }
Log 'starting step 7: DSEC-OWOD Task 1-4 (fusion add, base 10 epochs)'
& $py (Join-Path $root 'tools\train_owod.py') --fusion add --name owod_add
Log "step 7 finished (exit $LASTEXITCODE)"

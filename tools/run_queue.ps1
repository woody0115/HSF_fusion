# Run the experiment queue sequentially (own window, survives this session).
# Each line of the queue is: <name>|<python script and args>. Progress: runs\run_queue.log and the
# VS Code task "觀察訓練過程" (switches to each run automatically).
param([string]$Queue = 'step4_add|tools\train_dual.py --fusion add --epochs 20 --name dual_add_e20;step5_af|tools\train_dual.py --fusion af --epochs 20 --name dual_af_e20;step6_hsf|tools\train_dual.py --fusion hsf --epochs 20 --name dual_hsf_e20;step7_owod|tools\train_owod.py --fusion add --name owod_add')
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$log = Join-Path $root 'runs\run_queue.log'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $root
function Log($m) { $l = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $l; Add-Content $log $l -Encoding utf8 }

$pause = Join-Path $root 'runs\PAUSE'      # create this file to hold the queue after the current run
foreach ($item in $Queue.Split(';')) {
    $name, $cmd = $item.Split('|', 2)
    while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_single\.py|train_dual\.py|train_owod\.py' }) { Start-Sleep 60 }
    if (Test-Path $pause) { Log "paused before $name (delete runs\PAUSE to continue)" }
    while (Test-Path $pause) { Start-Sleep 30 }
    Log "start $name : $cmd"
    $argv = $cmd.Split(' ')
    $qlog = Join-Path $root "runs\queue_logs\$name.log"          # full output of every item
    New-Item -ItemType Directory -Force (Split-Path $qlog) | Out-Null
    cmd /c "$py $cmd > $qlog 2>&1"                  # paths here contain no spaces
    Log "end   $name (exit $LASTEXITCODE)"
}
Log 'queue finished'

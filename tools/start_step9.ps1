# Step 9 (event camera in the unknown decision): starts after the step-8 queue has finished.
#   1. option 1: event rescoring of the unknown score on the existing DSEC-OWOD run (owod_add)
#   2. option 3: retrain DSEC-OWOD with event-active background not taught as background,
#      only if option 1 showed an effect (train_owod.py --only-if)
#   3. option 1 evaluation on the option-3 run as well (skipped if option 3 was skipped)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'run_queue\.ps1' -and $_.CommandLine -match 'eval_iaf' }) { Start-Sleep 60 }
$q = 'step9_opt1|tools\owod_event_rescore.py --run runs\step7\owod_add;' +
     'step9_opt3|tools\train_owod.py --fusion add --event-ignore 0.1 --name owod_add_evtign --only-if runs\step9_evt_unknown\owod_add_rescore.json;' +
     'step9_opt3_rescore|tools\owod_event_rescore.py --run runs\step7\owod_add_evtign'
& (Join-Path $PSScriptRoot 'run_queue.ps1') -Queue $q

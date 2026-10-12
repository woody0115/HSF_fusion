# Live view of the DSEC download: current file, size, speed, ETA, and the log tail.
# Usage (VS Code: Terminal > Run Task > "觀察下載進度"):
#   powershell -ExecutionPolicy Bypass -File tools\watch_download.ps1 [-Root D:\DSEC]
param([string]$Root = 'D:\DSEC', [int]$Interval = 5)

# Expected sizes (GB) for ETA; same order as download_dsec_det.ps1
$expected = @{
    'train_object_detection_coarse__test_events.zip' = 6.94
    'test_object_detection_coarse__test_events.zip' = 6.94
    'test_object_detection_coarse__test_left_images_distorted.zip' = 6.38
    'train_object_detection_coarse__train_events.zip' = 20.40
    'train_object_detection_coarse__train_left_images_distorted.zip' = 27.68
    'test_coarse__test_events.zip' = 26.62
    'train_coarse__train_events.zip' = 122.83
}
$total = 211.0
$log = Join-Path $Root 'download.log'
$zips = Join-Path $Root '_zips'
$prev = @{}

while ($true) {
    $lines = @()
    $lines += "DSEC-Det download monitor    $(Get-Date -Format 'HH:mm:ss')   (Ctrl+C to stop watching; the download keeps running)"
    $lines += ''
    $active = Get-ChildItem $zips -Filter *.zip -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($active) {
        # pdownload.py pre-allocates the file, so progress comes from its chunk record (64 MB chunks)
        $bytes = $active.Length
        $progFile = "$($active.FullName).progress.json"
        if (Test-Path $progFile) {
            try { $bytes = [int64](Get-Content $progFile -Raw | ConvertFrom-Json).done.Count * 64MB } catch {}
        }
        $gb = $bytes / 1GB
        $speed = 0
        if ($prev.ContainsKey($active.Name)) { $speed = ($bytes - $prev[$active.Name]) / $Interval / 1MB }
        $prev[$active.Name] = $bytes
        $exp = $expected[$active.Name]
        $lines += "  current : $($active.Name)"
        if ($exp) {
            $pct = [math]::Min(100, $gb / $exp * 100)
            $bar = ('#' * [int]($pct / 4)).PadRight(25, '.')
            $eta = if ($speed -gt 0) { [timespan]::FromSeconds(($exp - $gb) * 1024 / $speed).ToString('hh\:mm\:ss') } else { '--:--:--' }
            $lines += ("  progress: [{0}] {1,5:N1}%   {2:N2} / {3:N2} GB" -f $bar, $pct, $gb, $exp)
            $lines += ("  speed   : {0:N1} MB/s    file ETA {1}" -f $speed, $eta)
        } else {
            $lines += ("  size    : {0:N2} GB    speed {1:N1} MB/s" -f $gb, $speed)
        }
    } else {
        $lines += '  (no archive downloading right now: extracting, finished, or not started)'
    }
    $doneFiles = (Get-ChildItem $zips -Filter *.done -ErrorAction SilentlyContinue).Count
    $free = (Get-PSDrive $Root.Substring(0, 1)).Free / 1GB
    $lines += ("  finished archives: {0} / 12      free on {1}: {2:N0} GB      (~{3:N0} GB total to fetch)" -f $doneFiles, $Root.Substring(0, 2), $free, $total)
    $lines += ''
    $lines += '  --- log ---'
    if (Test-Path $log) { $lines += Get-Content $log -Tail 12 | ForEach-Object { "  $_" } }
    Clear-Host
    $lines | ForEach-Object { Write-Host $_ }
    # stop only if the latest log line says the script ended (old FAILED lines from earlier runs are ignored)
    if ((Test-Path $log) -and ((Get-Content $log -Tail 1) -match 'ALL DONE|FAILED')) { Write-Host ''; Write-Host '  >>> download script finished (see log above)'; break }
    Start-Sleep $Interval
}

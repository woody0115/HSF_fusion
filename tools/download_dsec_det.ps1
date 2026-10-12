# Download only the DSEC parts needed for event-RGB object detection (DSEC-Det).
# Each archive is downloaded (resumable), extracted, then deleted to save space.
# Layout produced:  $Root\{train,test}\<sequence>\{events,images,calibration,object_detections}\...
# Usage:  powershell -ExecutionPolicy Bypass -File download_dsec_det.ps1 [-Root D:\DSEC]
param([string]$Root = 'D:\DSEC')

$ErrorActionPreference = 'Stop'
$base = 'https://download.ifi.uzh.ch/rpg/DSEC'
$zipDir = Join-Path $Root '_zips'
$log = Join-Path $Root 'download.log'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force $Root, $zipDir, "$Root\train", "$Root\test" | Out-Null

function Log($msg) {
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

# Small files first: labels + calibration, then the smallest complete events+RGB subset (test
# extra sequences) so preprocessing can be developed while the large archives download.
# D: has enough room for the largest zip plus its extracted copy at any point.
# Skipped on purpose: rectified RGB images (wrong view for the labels) and disparity / optical flow.
$files = @(
    @{ split = 'train'; rel = 'train_object_detection_coarse/train_object_detections.zip' },
    @{ split = 'test';  rel = 'test_object_detection_coarse/test_object_detections.zip' },
    @{ split = 'train'; rel = 'train_coarse/train_calibration.zip' },
    @{ split = 'test';  rel = 'test_coarse/test_calibration.zip' },
    @{ split = 'train'; rel = 'train_object_detection_coarse/train_calibration.zip' },
    @{ split = 'test';  rel = 'test_object_detection_coarse/test_calibration.zip' },
    @{ split = 'test';  rel = 'test_object_detection_coarse/test_events.zip' },                    #   6.9 GB  extra test sequences
    @{ split = 'test';  rel = 'test_object_detection_coarse/test_left_images_distorted.zip' },     #   6.4 GB  RGB remapped to event view
    @{ split = 'train'; rel = 'train_object_detection_coarse/train_events.zip' },                  #  20.4 GB  extra train sequences
    @{ split = 'train'; rel = 'train_object_detection_coarse/train_left_images_distorted.zip' },   #  27.7 GB
    @{ split = 'test';  rel = 'test_coarse/test_events.zip' },                                     #  26.6 GB
    @{ split = 'train'; rel = 'train_coarse/train_events.zip' }                                    # 122.8 GB
)

foreach ($f in $files) {
    $name = ($f.rel -replace '/', '__')                  # unique local name (two sources share file names)
    $zip = Join-Path $zipDir $name
    $done = Join-Path $zipDir "$name.done"
    if (Test-Path $done) { Log "skip (already done): $($f.rel)"; continue }

    # 1) download with resume; the server throttles single connections (~1 MB/s), so use 4 parallel ones
    Log "download: $($f.rel)"
    for ($try = 1; $try -le 5; $try++) {
        & $python "$PSScriptRoot\pdownload.py" "$base/$($f.rel)" $zip --workers 4
        if ($LASTEXITCODE -eq 0) { break }
        Log "  downloader exit $LASTEXITCODE, retry $try"
        Start-Sleep 30
    }
    if ($LASTEXITCODE -ne 0) { Log "FAILED download: $($f.rel)"; exit 1 }
    Log ("  size {0:N2} GB" -f ((Get-Item $zip).Length / 1GB))

    # 2) extract: archives either contain '<split>/...' paths or bare sequence folders
    $first = (& tar.exe -tf $zip | Select-Object -First 1)
    $dest = if ($first -match '^(train|test)/') { $Root } else { Join-Path $Root $f.split }
    Log "  extract to $dest (first entry: $first)"
    & tar.exe -xf $zip -C $dest
    if ($LASTEXITCODE -ne 0) { Log "FAILED extract: $($f.rel)"; exit 1 }

    # 3) free the space
    Remove-Item $zip -Force
    New-Item -ItemType File $done | Out-Null
    $free = (Get-PSDrive ($Root.Substring(0, 1))).Free / 1GB
    Log ("  done; free space on drive: {0:N0} GB" -f $free)
}
Log 'ALL DONE'

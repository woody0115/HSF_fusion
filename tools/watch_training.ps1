# Live view of the most recent training run (the path is written to runs\latest_log.txt by the
# training scripts). Shows the log as it grows; switches automatically when a new run starts.
# VS Code: Terminal > Run Task > "觀察訓練過程"   (Ctrl+C to stop watching; training keeps running)
$root = Split-Path $PSScriptRoot -Parent
$pointer = Join-Path $root 'runs\latest_log.txt'
$current = ''
$pos = 0
Write-Host 'waiting for a training run...'
while ($true) {
    if (Test-Path $pointer) {
        $log = (Get-Content $pointer -Raw).Trim()
        if ($log -ne $current -and (Test-Path $log)) {
            $current = $log; $pos = 0
            Write-Host ''
            Write-Host "===== $current =====" -ForegroundColor Cyan
        }
        if ($current -and (Test-Path $current)) {
            $fs = [System.IO.File]::Open($current, 'Open', 'Read', 'ReadWrite')
            if ($fs.Length -gt $pos) {
                $fs.Seek($pos, 'Begin') | Out-Null
                $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
                $text = $sr.ReadToEnd()
                $pos = $fs.Length
                $sr.Close()
                # collapse progress-bar spam: keep only the last progress line of each chunk
                $lines = $text -split "`n" | Where-Object { $_.Trim() }
                $prog = $lines | Where-Object { $_ -match '\d+/\d+ .*(it/s|s/it)' } | Select-Object -Last 1
                $lines | Where-Object { $_ -notmatch '\d+/\d+ .*(it/s|s/it)' } | ForEach-Object { Write-Host $_ }
                if ($prog) { Write-Host $prog -ForegroundColor DarkGray }
            } else { $fs.Close() }
        }
    }
    Start-Sleep 2
}

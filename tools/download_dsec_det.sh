#!/usr/bin/env bash
# Linux version of download_dsec_det.ps1: download only the DSEC parts needed for event-RGB object
# detection (DSEC-Det). Each archive is downloaded (resumable), extracted, then deleted to save space.
# Layout produced:  $ROOT/{train,test}/<sequence>/{events,images,calibration,object_detections}/...
# Usage:  bash tools/download_dsec_det.sh [ROOT]          (default ~/work/shared_data/DSEC)
#         nohup bash tools/download_dsec_det.sh > download.out 2>&1 &   (keeps running after the terminal closes)
set -u

ROOT=${1:-$HOME/work/shared_data/DSEC}
BASE=https://download.ifi.uzh.ch/rpg/DSEC
ZIPS=$ROOT/_zips
LOG=$ROOT/download.log
PYTHON=${PYTHON:-python3}                 # pdownload.py only needs the standard library
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT" "$ZIPS" "$ROOT/train" "$ROOT/test"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

# Same order as the Windows script: small files first, then the archives from small to large.
# Skipped on purpose: rectified RGB images (wrong view for the labels) and disparity / optical flow.
FILES=(
  "train train_object_detection_coarse/train_object_detections.zip"
  "test  test_object_detection_coarse/test_object_detections.zip"
  "train train_coarse/train_calibration.zip"
  "test  test_coarse/test_calibration.zip"
  "train train_object_detection_coarse/train_calibration.zip"
  "test  test_object_detection_coarse/test_calibration.zip"
  "test  test_object_detection_coarse/test_events.zip"                   #   6.9 GB  extra test sequences
  "test  test_object_detection_coarse/test_left_images_distorted.zip"    #   6.4 GB  RGB remapped to event view
  "train train_object_detection_coarse/train_events.zip"                 #  20.4 GB  extra train sequences
  "train train_object_detection_coarse/train_left_images_distorted.zip"  #  27.7 GB
  "test  test_coarse/test_events.zip"                                    #  26.6 GB
  "train train_coarse/train_events.zip"                                  # 122.8 GB
)

for entry in "${FILES[@]}"; do
  read -r split rel <<< "$entry"
  name=${rel//\//__}                      # unique local name (two sources share file names)
  zip=$ZIPS/$name
  if [ -f "$zip.done" ]; then log "skip (already done): $rel"; continue; fi

  # 1) download with resume; the server throttles single connections, so use 4 parallel ones
  log "download: $rel"
  ok=0
  for try in 1 2 3 4 5; do
    if "$PYTHON" "$HERE/pdownload.py" "$BASE/$rel" "$zip" --workers 4; then ok=1; break; fi
    log "  downloader failed, retry $try"
    sleep 30
  done
  if [ $ok -ne 1 ]; then log "FAILED download: $rel"; exit 1; fi
  log "  size $(du -h "$zip" | cut -f1)"

  # 2) extract: archives either contain '<split>/...' paths or bare sequence folders
  first=$(unzip -Z1 "$zip" | head -1)
  case $first in train/*|test/*) dest=$ROOT ;; *) dest=$ROOT/$split ;; esac
  log "  extract to $dest (first entry: $first)"
  if ! unzip -q -o "$zip" -d "$dest"; then log "FAILED extract: $rel"; exit 1; fi

  # 3) free the space
  rm -f "$zip"
  touch "$zip.done"
  log "  done; free space: $(df -h "$ROOT" | awk 'NR==2 {print $4}')"
done
log 'ALL DONE'

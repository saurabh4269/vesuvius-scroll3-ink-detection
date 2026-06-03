#!/bin/bash
# Login-node prep: download a cheap coarse internal level (~9.6um/px) of every
# m7 ink-prediction zarr in the open-data bucket, for the SLURM triage job to read.
set -u
B=s3://vesuvius-challenge-open-data
DEST="$HOME/scroll_prize/data/m7_triage"
mkdir -p "$DEST"
scrolls=$(aws s3 ls $B/ --no-sign-request | awk '/PRE/{print $NF}' | tr -d '/' | grep -v thumbnail)
n=0
for s in $scrolls; do
  base="$B/$s/representations/predictions/surfaces/"
  zarr=$(aws s3 ls "$base" --no-sign-request 2>/dev/null | awk '/PRE/{print $NF}' | grep 'surface-m7' | head -1 | tr -d '/')
  [ -z "$zarr" ] && continue
  # prefer internal level 3 (~9.6um), fall back to 2 then 4
  lvl=""
  for cand in 3 2 4; do
    if aws s3 ls "$base$zarr/$cand/.zarray" --no-sign-request >/dev/null 2>&1; then lvl=$cand; break; fi
  done
  [ -z "$lvl" ] && { echo "SKIP $s (no coarse level)"; continue; }
  if [ -f "$DEST/$s/L3/.zarray" ]; then echo "HAVE $s"; n=$((n+1)); continue; fi
  echo "SYNC $s  zarr=$zarr  level=$lvl"
  aws s3 sync "$base$zarr/$lvl/" "$DEST/$s/L3/" --no-sign-request --quiet && n=$((n+1))
done
echo "PREP DONE: $n scrolls ready"
du -sh "$DEST"/*/ 2>/dev/null

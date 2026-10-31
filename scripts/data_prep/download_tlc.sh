#!/bin/bash
# Download NYC TLC trip records (Uber/Lyft "fhvhv" and yellow cabs) for the given months.
# usage: scripts/data_prep/download_tlc.sh 2024-01 2024-02 ...   (into data/history/raw/, or $OUT_DIR)
cd "$(dirname "$0")/../.." || exit 1
OUT_DIR=${OUT_DIR:-data/history/raw}
mkdir -p "$OUT_DIR"
for month in "$@"; do
  for kind in fhvhv yellow; do
    file="$OUT_DIR/${kind}_tripdata_${month}.parquet"
    [ -s "$file" ] && continue
    curl -sSf --retry 3 -o "$file.part" "https://d37ci6vzurychx.cloudfront.net/trip-data/${kind}_tripdata_${month}.parquet" \
      && mv "$file.part" "$file" && echo "$file" || { echo "failed: $file"; rm -f "$file.part"; }
  done
done

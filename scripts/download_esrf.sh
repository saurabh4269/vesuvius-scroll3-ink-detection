#!/bin/bash
# Download ESRF fragment surface volumes + labels for MiniUNETR training.
# Run as: nohup bash download_esrf.sh > ~/logs/download_esrf.log 2>&1 &

source ~/miniconda3/etc/profile.d/conda.sh && conda activate scroll

BASE="https://dl.ash2txt.org/fragments"
DATA="$HOME/scroll_prize/data/esrf/scroll0/fragments"

for FRAG in 500P2 343P; do
    FRAG_URL="${BASE}/PHerc0${FRAG}/paths/2um_front_surface"
    DEST="${DATA}/${FRAG}"
    mkdir -p "${DEST}/layers"

    echo "[$(date)] === Downloading ${FRAG} ==="

    # Ink label
    if [ ! -f "${DEST}/${FRAG}_inklabels.png" ]; then
        wget -q "${FRAG_URL}/${FRAG}_inklabels.png" -O "${DEST}/${FRAG}_inklabels.png" \
            && echo "  inklabels OK" || echo "  inklabels FAILED"
    else
        echo "  inklabels already exists"
    fi

    # Mask
    if [ ! -f "${DEST}/${FRAG}_mask.png" ]; then
        wget -q "${FRAG_URL}/${FRAG}_mask.png" -O "${DEST}/${FRAG}_mask.png" \
            && echo "  mask OK" || echo "  mask FAILED"
    else
        echo "  mask already exists"
    fi

    # 66 surface layers (00.png..65.png)
    echo "  Downloading 66 layers..."
    for i in $(seq 0 65); do
        layer=$(printf "%02d" $i)
        dest_file="${DEST}/layers/${layer}.png"
        if [ ! -f "${dest_file}" ] || [ ! -s "${dest_file}" ]; then
            wget -q "${FRAG_URL}/layers/${layer}.png" -O "${dest_file}" \
                || echo "  Layer ${layer} FAILED"
        fi
    done
    COUNT=$(ls "${DEST}/layers/"*.png 2>/dev/null | wc -l)
    echo "  [$(date)] ${FRAG} layers: ${COUNT}/66. Size: $(du -sh ${DEST})"
done

echo "[$(date)] All ESRF downloads done."
du -sh "${DATA}"

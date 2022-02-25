#!/usr/bin/env bash

python tools/train.py configs/beit/upernet/upernet_beit_large_24_512_slide_160k_potsdam_ms.py \
    --work-dir experiments/rpe-l-gid-500 --seed 0  --deterministic \
    --options model.pretrained=/root/beitpre-d/work/beit-rpe-gid-l-500/checkpoint-499.pth
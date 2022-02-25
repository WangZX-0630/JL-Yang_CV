#!/usr/bin/env bash

python tools/train.py configs/beit/upernet/upernet_beit_base_12_512_slide_160k_gid_lepe.py \
    --work-dir experiments/b-scratch --seed 0  --deterministic \
    --resume-from experiments/b-scratch/iter_130000.pth
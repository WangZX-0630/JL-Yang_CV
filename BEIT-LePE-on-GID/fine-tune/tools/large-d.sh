#!/usr/bin/env bash

PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    tools/train.py configs/beit/upernet/upernet_beit_large_24_512_slide_160k_potsdam_ms-dist.py \
     --work-dir experiments/rpe-l-gid-500 --seed 0  --deterministic \
    --options model.pretrained=/root/beitpre-d/work/beit-rpe-gid-l-500/checkpoint-499.pth \
    --launcher pytorch ${@:3}

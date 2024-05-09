#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
PORT=${PORT:-29500}


PYTHONPATH=/mnt/diskg/ruitong_gan/gaiaseg_huawei_pack/GAIA-cv-master:"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/count_flops.py \
    ${CONFIG} \
    --launcher pytorch \
    --work-dir ${WORK_DIR}


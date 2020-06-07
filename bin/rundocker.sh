#!/bin/bash

SRCDIR="$(dirname $0)/.."
SRCDIR="$(realpath $SRCDIR)"
DATADIR="$(dirname $0)/../../data"
DATADIR="$(realpath $DATADIR)"


docker run -ti \
    --name "narrative-extraction-$RANDOM" \
    --rm \
    --gpus=all \
    --ipc=host \
    -v "$SRCDIR":/workspace \
    -v "$DATADIR":/data \
    -v "$DATADIR/deeppavlov_workdir":/root/.deeppavlov \
    -e TORCH_HOME=/data/pretrained_models \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    rsuvorov/narrative-extraction \
    $@

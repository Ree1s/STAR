#!/bin/bash
# Set HIP_VISIBLE_DEVICES to specify which GPU(s) to use
# export HSA_OVERRIDE_GFX_VERSION=10.3.0
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set MIOpen cache and database paths to avoid SQLite errors
export MIOPEN_USER_DB_PATH="/group/ossdphi_algo_scratch_14/sichegao/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Clear and recreate the MIOpen cache directory
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

torchrun --nnodes=1 --nproc_per_node=4 scripts/train.py --config configs/mvdit/train/16x256x256.py
# export RANK=0
# export WORLD_SIZE=1
# export LOCAL_RANK=0
# export MASTER_ADDR='localhost'
# export MASTER_PORT='29500'
# python scripts/train.py --config configs/mvdit/train/16x256x256.py

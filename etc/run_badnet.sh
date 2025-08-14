#!/bin/bash

POISON_RATE=$1
CUDA_VISIBLE_DEVICES=3 python python/train_backdoor.py \
  --data-set ${DATASET} --input-size 32 --data-path ~/data \
  --verbose 0 \
  --model ${MODEL} \
  --epochs 300 --opt momentum --momentum 0.9  --lr 0.01 --sched cosine  \
  --no-pin-mem  \
  --no-model-ema \
  --aug-method simple \
  --batch-size 128 \
  --selection_strategy ${STRAG} --num_workers 8 \
  --surrogate_model ${SURR_MODEL} \
  --clean-acc-tol 0.1 --attack_portion ${POISON_RATE} --attack_mode 'clean_label' --attack_label ${LABEL}  --attack_pixel_val 1 --attack_pattern_width 3 --attack_type badnet --badnet_trigger hard \
  --output_dir ${base_dir} \
  --external_logger none --external_logger_args backdoor --run_name badnet_${POISON_RATE}_${DATASET}_${STRAG} --save_ckpt \


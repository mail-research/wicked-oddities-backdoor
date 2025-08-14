#!/bin/bash
POISON_RATE=$1
CUDA_VISIBLE_DEVICES=0 python python/train_backdoor.py \
  --data-set ${DATASET} --input-size 32 --data-path ~/data \
  --verbose 2 \
  --model ${MODEL} \
  --epochs 300 --opt momentum --momentum 0.9  --lr 0.01 \
  --no-pin-mem \
  --no-model-ema \
  --aug-method simple \
  --batch-size 128 --num_workers 8 \
  --selection_strategy ${STRAG} \
  --surrogate_model ${SURR_MODEL} \
  --k ${K} --subset_size ${SUBSET_SIZE} \
  --clean-acc-tol 0.1 --attack_portion ${POISON_RATE} --attack_mode 'clean_label' --attack_label ${LABEL} --attack_type SIG \
  --output_dir ${base_dir} \
  --external_logger wandb --external_logger_args backdoor --run_name SIG_${POISON_RATE}_${DATASET}_${STRAG} --save_ckpt \


#!/bin/bash

python ../src/main.py \
  --num_episodes 500 \
  --num_hunters 6 \
  --num_targets 2 \
  --save_frequency 100 \
  --ifrender false \
  --visualizelaser false \
  --update_freq 10 \
  --batch_size 256 \
  --lr 5e-4 
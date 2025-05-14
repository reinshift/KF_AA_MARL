@echo off

set MODEL_DIR=..\model\20241223_134914_score_557

python ..\src\test_model.py ^
  --model_dir %MODEL_DIR% ^
  --num_hunters 6 ^
  --num_targets 2 ^
  --max_steps 150 ^
  --num_test_episodes 1 ^
  --seed 10 ^
  --ifrender true ^
  --visualizelaser false
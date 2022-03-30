# pre-train, motion prediction
# joint data, sequence length 150, final_size=2048, 8 layers
CUDA_VISIBLE_DEVICES=0 python run_motion.py \
  --use_seq_mean=True \
  --is_finetune=False \
  --freeze_vars=False \
  --input_file='input_dir'/tf_train.tfrecord \
  --output_dir='output_dir' \
  --do_train=True \
  --do_eval=False \
  --do_predict=False \
  --bert_config_file=./config/L8_2048.json \
  --train_batch_size=16 \
  --eval_batch_size=64 \
  --max_seq_length=150 \
  --max_predictions_per_seq=22 \
  --num_train_steps=200000 \
  --num_warmup_steps=20000 \
  --learning_rate=1e-4 \
  --length_feature=75 \
  --model=modeling_hierarchical \
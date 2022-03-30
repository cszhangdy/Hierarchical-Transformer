# finetuning for classification
# joint data, sequence length 150, final_size=2048, 4 layers
CUDA_VISIBLE_DEVICES=5 python run_finetune_flow_predict.py \
  --use_seq_mean=True \
  --is_finetune=True \
  --freeze_vars=True \
  --input_file='input_dir'/tf_finetune_train.tfrecord \
  --output_dir='output_dir' \
  --init_checkpoint='init_dir' \ # pre-training model dir
  --do_train=True \
  --do_eval=False \
  --do_predict=False \
  --bert_config_file=./data_cloud/L4_2048.json \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --max_seq_length=150 \
  --max_predictions_per_seq=22 \
  --num_train_steps=50000 \
  --num_warmup_steps=5000 \
  --learning_rate=1e-3 \
  --length_feature=75 \
  --model=modeling_hierarchical \
  #--use_hinge_loss=True \

# try predict motion as classification
# use NTU60 data, cross-subject setting
python create_training_data_for_motion.py \
  --data_dir='dir_to_ntu_xsub' \
  --output_file_train='output_dir'/tf_train.tfrecord \
  --output_file_valid='output_dir'/tf_valid.tfrecord \
  --output_file_finetune='output_dir'/tf_finetune.tfrecord \
  --max_seq_length=150 \
  --max_predictions_per_seq=22 \
  --sub_seq_num=150 \
  --masked_lm_prob=0.15 \
  --dupe_factor=1 \
  --length_feature=75 \
  --mask_length=1 \
  --data_struc=joint \
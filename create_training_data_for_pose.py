""" Sampling MaxT frames, use only first person. remove [CLS] and [SEP]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import h5py
import numpy as np
import collections
import tensorflow as tf
import glob
import os
import copy
import pickle
from tqdm import tqdm
import utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_file_train", None,
    "Output TF example file for trian (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file_valid", None,
    "Output TF example file for valid (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file_finetune", None,
    "Output TF example file for valid (or comma-separated list of files).")

flags.DEFINE_string("data_dir", None,
                    "The 3D human pose of time series.")

flags.DEFINE_string("data_struc", None,
                    "choose bone or joint")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer(
    "dupe_factor", 2,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_integer(
    "downsample_factor", 1,
    "Number of times to downsampling the input data.")

flags.DEFINE_integer(
    "mask_length", 4,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_integer(
    "sub_seq_num", 64,
    "Number of sampling number of frames.")

flags.DEFINE_integer(
    "length_feature", 75,
    "Dimension of the 3d pose vector.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")



def read_data(data_dir, mode, data_struc):

  ### 3d pose error:
  # two person interaction 6889(288), 3458(167)
  # single person interaction 30757(1226), 15474(573)
  # 2019.11.14: Interaction is noisy, just use the first person in every skeleton file
  # two_person_actions = list(range(49,60)) # + list(range(105,120))

  # if mode == 'val':
  #   data_dir = '/data/chengyibin/motion_3d/datasets/ntu60-bert/xsub'
  #   print(data_dir)
  
  # 'train', 'val'
  # 'joint', 'bone'

  data_path = os.path.join(data_dir, '{}_data_{}.npy'.format(mode, data_struc))
  label_path = os.path.join(data_dir, '{}_label.pkl'.format(mode))

  # data: (N,C,V,T,M)
  try:
      with open(label_path, 'rb') as f:
          # sample_names, labels = pickle.load(f)
          sample_names, labels, sample_length = pickle.load(f)
  except:
      # for pickle file from python2
      with open(label_path, 'rb') as f:
          sample_names, labels = pickle.load(f)
          setting = 'xsub' if 'xsub' in data_dir else 'xview'
          with open(f'/data/chengyibin/motion_3d/datasets/ntu60-bert/{setting}/{mode}_label.pkl', 'rb') as f:
            _, _, sample_length = pickle.load(f)
          print (f'load data from {data_dir}')

  data = np.load(data_path, mmap_mode='r')

  N, C, T, V, M = data.shape

  is_interaction = []
  is_exchange = []
  output = []
  label_output = []

  def normalize_video(video):
    """ 0805
    :param video: np array of shape [seq_len, coordinate]
    :return:
    """
    max_75 = np.amax(video, axis=0)
    min_75 = np.amin(video, axis=0)
    max_x = np.max([max_75[i] for i in range(0,75,3)])
    max_y = np.max([max_75[i] for i in range(1,75,3)])
    max_z = np.max([max_75[i] for i in range(2,75,3)])
    min_x = np.min([min_75[i] for i in range(0,75,3)])
    min_y = np.min([min_75[i] for i in range(1,75,3)])
    min_z = np.min([min_75[i] for i in range(2,75,3)])
    norm = np.zeros_like(video)
    for i in range(0,75,3):
        norm[:,i] = 2*(video[:,i]-min_x)/(max_x-min_x)-1
        norm[:,i+1] = 2*(video[:,i+1]-min_y)/(max_y-min_y)-1
        norm[:,i+2] = 2*(video[:,i+2]-min_z)/(max_z-min_z)-1
        # if max_x - min_x > 0:
        #   norm[:,i] = 2*(video[:,i]-min_x)/(max_x-min_x)-1
        # if max_y - min_y > 0:
        #   norm[:,i+1] = 2*(video[:,i+1]-min_y)/(max_y-min_y)-1
        # if max_z - min_z > 0:
        #   norm[:,i+2] = 2*(video[:,i+2]-min_z)/(max_z-min_z)-1
    return norm

  maxT = FLAGS.sub_seq_num ######
  for seq, label, frame_num in tqdm(zip(data, labels, sample_length)):
      # M, T, V, C
      # seq = seq.transpose((3,1,2,0)).reshape((M, T, -1))[0]  # only use first person
      # # Sample maxT frames
      # if frame_num <= maxT:
      #     seq = seq[:maxT, :]
      # else:
      #     ## sample 'self.maxT' frames
      #     s = frame_num // maxT
      #     seq = seq[::s, :][:maxT, :]
      #     # r = np.random.randint(0, frame_num - self.maxT * s + 1) # [low, high)
      #     # r = np.random.randint(0, max(frame_num - maxT*s, s)) # [low, high)
      #     # seq = seq[:, r::s, :][:, :maxT, :]

      # 0808f, gen 150 data
      seq = seq.transpose((3,1,2,0)).reshape((M, T, -1))[0]  # only use first person
      # normalize first then downsample
      seq = normalize_video(seq)
      seq = seq[::2]

      # print (seq.shape)
      # input()
      output.append(seq) # (maxT, 75)
      label_output.append(label)
      is_interaction.append(0)
      is_exchange.append(0)
      
      # print(output[-1].shape)
      # if len(output) > 10:
      #   # input()
      #   break

  return output, is_interaction, is_exchange, label_output


def padding_for_normalization(train_seqs):
  padding_num = np.asarray([FLAGS.max_seq_length-3-len(e) for e in train_seqs]).sum()
  train_seqs.append(np.zeros((padding_num, FLAGS.length_feature), dtype=np.float32))
  return train_seqs

def cut_fix_length(seq, seq_index, repeat_idx, mode):
 
  window_lenth = FLAGS.max_seq_length-2 # for only one person
  if len(seq) > window_lenth:
    if 'finetune' in mode:
      if repeat_idx == 0:
  	    start = 0
      else:
        start = len(seq) - window_lenth
    else:
      start = random.randint(0, len(seq) - window_lenth)
    select_index = range(start, start+window_lenth)
    return seq[select_index], seq_index[select_index]
  else:
    return seq, seq_index


def generate_masked_sample(input_, is_interaction, is_exchange, mode): 
  output_ = [e for e in input_.copy()]   
  masked_lm_position = []
  masked_lm_ids = []
  masked_lm_weights = []
  input_mask = [1 for _ in input_]
  segment_ids = [0 for _ in range(len(input_)//2)] + [1 for _ in range(len(input_)//2)]
  
  # if 'finetune' not in mode:

  #   if is_interaction:
  #     input_mask_A = [1 for _ in range(len(input_)//2)] + [0 for _ in range(len(input_)//2)]
  #     input_mask_B = [0 for _ in range(len(input_)//2)] + [1 for _ in range(len(input_)//2)]
  #   else:
  #     if not is_exchange:
  #       input_mask_A = [0 for _ in range(len(input_)//2)] + [1 for _ in range(len(input_)//2)]
  #       input_mask_B = [0 for _ in range(len(input_)//2)] + [1 for _ in range(len(input_)//2)]
  #     else:
  #       input_mask_A = [1 for _ in range(len(input_)//2)] + [0 for _ in range(len(input_)//2)]
  #       input_mask_B = [1 for _ in range(len(input_)//2)] + [0 for _ in range(len(input_)//2)]

  # else:
  #     input_mask_A = [0 for _ in range(len(input_)//2)] + [0 for _ in range(len(input_)//2)]
  #     input_mask_B = [0 for _ in range(len(input_)//2)] + [0 for _ in range(len(input_)//2)]

  input_mask_A = input_mask.copy()
  input_mask_B = input_mask.copy()


  if 'finetune' not in mode and len(input_)!=0:    
    factor = FLAGS.mask_length

    select_num = int(len(input_)*FLAGS.masked_lm_prob)//factor if int(len(input_)*FLAGS.masked_lm_prob)//factor!=0 else 1

    try:
      masked_index = random.sample(range((len(input_))//factor), select_num)
    except:
      print(range((len(input_))//factor), select_num)

    for i in masked_index:
      masked_lm_position += list(range(factor*i, factor*(i+1)))

    for ele in masked_lm_position:
      masked_frame = None
      if random.random() < 0.8: # 80% of the time, replace with
        masked_frame = np.zeros(FLAGS.length_feature, dtype=np.float32)
      else:
        if random.random() < 0.5: # 10% of the time, keep original
          masked_frame = output_[ele]
        else: # 10% of the time, replace with random word
          masked_frame = output_[random.randint(0, len(output_) - 1)]

      masked_lm_ids.append(output_[ele])
      output_[ele] = masked_frame
    
    masked_lm_weights = [1 for _ in masked_lm_ids]

  return output_, segment_ids, masked_lm_position, input_mask, input_mask_A, input_mask_B, masked_lm_ids, masked_lm_weights


def add_CLS_SEP(sub_seq_, segment_ids, masked_lm_position, input_mask, input_mask_A, input_mask_B):

  length_init = len(sub_seq_)

  sub_seq_.insert(0, -1.0*np.ones(FLAGS.length_feature, dtype = np.float32))
  sub_seq_.insert(length_init//2+1, 1.0*np.ones(FLAGS.length_feature, dtype = np.float32))
  sub_seq_.append(np.ones(FLAGS.length_feature, dtype = np.float32))

  segment_ids.insert(0, 0)
  segment_ids.insert(length_init//2+1, 0)
  segment_ids.append(1)

  for i in range(len(masked_lm_position)):
  	if masked_lm_position[i] >= length_init//2:
  		masked_lm_position[i] += 1
  masked_lm_position = [e+1 for e in masked_lm_position]

  input_mask += [1, 1, 1]

  # print(input_mask)

  input_mask_A.insert(0, 0)
  input_mask_A.append(0)
  input_mask_A.append(0)

  input_mask_B.insert(0, 0)
  input_mask_B.insert(0, 0)
  input_mask_B.append(0)

  return sub_seq_, segment_ids, masked_lm_position, input_mask, input_mask_A, input_mask_B


def padding(data):
  #[sub_seq_, input_mask, input_mask_A, input_mask_B, segment_ids, masked_lm_position, masked_lm_ids, masked_lm_weights, sub_seq_index_, action_labels, total_index]	
  #     0          1           2            3               4                5                6         7                     8                 9            10
  data[0] += [np.zeros(FLAGS.length_feature, dtype=np.float32) for _ in range(FLAGS.max_seq_length-len(data[0]))]
  data[1] += [0 for _ in range(FLAGS.max_seq_length-len(data[1]))]
  data[2] += [0 for _ in range(FLAGS.max_seq_length-len(data[2]))]
  data[3] += [0 for _ in range(FLAGS.max_seq_length-len(data[3]))]
  data[4] += [0 for _ in range(FLAGS.max_seq_length-len(data[4]))]
  data[5] += [0 for _ in range(FLAGS.max_predictions_per_seq-len(data[5]))]
  data[6] += [np.zeros(FLAGS.length_feature, dtype=np.float32) for _ in range(FLAGS.max_predictions_per_seq-len(data[6]))]
  data[7] += [0 for _ in range(FLAGS.max_predictions_per_seq-len(data[7]))]
  data[8] += [0 for _ in range(FLAGS.max_seq_length-2-len(data[8]))]


  return data


def save_to_tfrecorder(data, writer):

  # [print(idx, '\n', ele[:20]) for idx, ele in enumerate(data[0])]
  # print(np.asarray(data[1]).sum())

  # print(len(data[0]))
  # print(len(data[1]))
  # input()

  def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  # print([e.shape for e in data[0]])

  data[0] = np.stack(data[0]).astype(np.float32).reshape((-1))
  data[1] = np.stack(data[1]).astype(np.int64)

  ## get maskA and maskB
  data[2] = (1 - np.stack(data[2])).astype(np.int64)
  data[3] = (1 - np.stack(data[3])).astype(np.int64)
  data[4] = np.stack(data[4]).astype(np.int64)
  data[5] = np.stack(data[5]).astype(np.int64)
  data[6] = np.stack(data[6]).astype(np.float32).reshape((-1))
  data[7] = np.stack(data[7]).astype(np.float32)
  data[8] = np.stack(data[8]).astype(np.float32)

  # if len(data[8])!=126:
  #   print(len(data[8]))

  ## the shape of the input_mask and masked_lm_ids is [number_of_samples, 42]
  feature_dict = {
      'input_ids': float_feature(data[0]),
      'input_mask': int64_feature(data[1]),
      'input_mask_A': int64_feature(data[2]),
      'input_mask_B': int64_feature(data[3]),
      'segment_ids': int64_feature(data[4]),
      'masked_lm_positions': int64_feature(data[5]),
      'masked_lm_ids': float_feature(data[6]),
      'masked_lm_weights': float_feature(data[7]),
      'sub_seq_index': float_feature(data[8]),
      'action_labels': int64_feature([data[9]]),
      'total_index': int64_feature([data[10]]),
      }

  tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  writer.write(tf_example.SerializeToString())
  return writer

def create_data_for_tfrecorder(seqs, is_interaction, is_exchange, labels, mode):

  if mode == 'train':
    name = FLAGS.output_file_train
  elif mode == 'eval':
    name = FLAGS.output_file_valid
  elif 'finetune' in mode:
    name = FLAGS.output_file_finetune
    name = name.replace("tf_finetune.tfrecord", "tf_{}.tfrecord".format(mode))

  print(name)
  writer = tf.python_io.TFRecordWriter(name)

  count = 0

  for idx, seq in tqdm(enumerate(seqs)):

    label = labels[idx]

    seq_index = np.asarray(list(range(0, len(seq))))

    # if 'eval' in mode or 'finetune' in mode: ## create validation data, add eval option
    if 'finetune' in mode:
      repeat_num = 1
    else:
      repeat_num = 2

    for repeat_idx in range(repeat_num):
      seq_ = seq.copy()
      seq_index_ = seq_index.copy()

      # cut 
      # sub_seq_, sub_seq_index_ = cut_fix_length(seq_, seq_index_, repeat_idx, mode)
      sub_seq_, sub_seq_index_ = seq_, seq_index_
      
      # mask
      sub_seq_, segment_ids, masked_lm_position, input_mask, \
           input_mask_A, input_mask_B, masked_lm_ids, masked_lm_weights = generate_masked_sample(sub_seq_, is_interaction, is_exchange, mode)


      # add [CLS] [SEP]
      # sub_seq_, segment_ids, masked_lm_position, input_mask, input_mask_A, input_mask_B\
      #       = add_CLS_SEP(sub_seq_, segment_ids, masked_lm_position, input_mask, input_mask_A, input_mask_B)

      # print(input_mask)      

      # group
      data_to_write = [sub_seq_, input_mask, input_mask_A, input_mask_B, segment_ids, masked_lm_position, \
                       masked_lm_ids, masked_lm_weights, list(sub_seq_index_), label, 0]

      # print(data_to_write[1])


      # padding
      data_to_write = padding(data_to_write)

      # ## save data to tf_recoder
      writer = save_to_tfrecorder(data_to_write, writer)
      count += 1
      
      if count%1000 == 0:
        print('process {} data'.format(count))

      # if count > 6:
      #   break

  print('Totally prepare {} sequences'.format(count))
  writer.close()

  return 

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.logging.info("*** Reading from input files ***")

  # train_seqs, test_seqs = read_data(FLAGS.sbu_dir)
  train_seqs = None
  train_seqs, is_interaction_train, is_exchange_train, labels_train = read_data(FLAGS.data_dir, 'train', FLAGS.data_struc)

  valid_seqs = None
  valid_seqs, is_interaction_valid, is_exchange_valid, labels_valid = read_data(FLAGS.data_dir, 'val', FLAGS.data_struc)

  # train_seqs, valid_seqs = utils.normalize_data(train_seqs, valid_seqs, f'mean-std-h5/mean_std_ori_joint_1_seq{FLAGS.max_seq_length}.h5')
  # train_seqs, valid_seqs = utils.normalize_data(train_seqs, valid_seqs)

  tf.logging.info("*** Writing training data to output files for train ***")
  create_data_for_tfrecorder(train_seqs, is_interaction_train, is_exchange_train, labels_train, mode='train')

  #tf.logging.info("*** Writing validation data to output files for valid ***")
  #create_data_for_tfrecorder(valid_seqs, is_interaction_valid, is_exchange_valid, labels_valid, mode='eval')

  tf.logging.info("*** Writing no masked data to output files for finetune ***")
  create_data_for_tfrecorder(train_seqs, is_interaction_train, is_exchange_train, labels_train, mode='finetune_train')
  tf.logging.info("*** Writing no masked data to output files for finetune evaluation***")
  create_data_for_tfrecorder(valid_seqs, is_interaction_valid, is_exchange_valid, labels_valid, mode='finetune_valid')

if __name__ == "__main__":
  tf.app.run()

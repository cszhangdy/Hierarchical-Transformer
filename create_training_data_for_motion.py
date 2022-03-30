from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random
import numpy as np
import tensorflow as tf
import os
import copy
import pickle
from tqdm import tqdm

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
    "sub_seq_num", 64,
    "Number of frames for each person.")

flags.DEFINE_integer(
    "downsample_factor", 1,
    "Number of times to downsampling the input data.")

flags.DEFINE_integer(
    "mask_length", 1,
    "Number of consecutive masked frames.")

flags.DEFINE_integer(
    "length_feature", 75,
    "Dimension of the 3d pose vector.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")



def read_data(data_dir, mode, data_struc):
  ### NTU joint indices to SUB
  ntu_to_sbu = np.array([3, 20, 1, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 0])
  sbu_to_ntu = np.array([[2, 6, 15, 19, 6, 21, 22, 10, 23, 24],
                        [1, 5, 11, 14, 5, 5,  5,  8,  8,  8]])

  data_path = os.path.join(data_dir, '{}_data_{}.npy'.format(mode, data_struc))
  label_path = os.path.join(data_dir, '{}_label.pkl'.format(mode))


  data = np.load(data_path, mmap_mode='r')
  # data.shape: (N,C,V,T,M)
  sample_length = None
  try:
    with open(label_path, 'rb') as f:
      sample_names, labels = pickle.load(f)
      sample_length = data.shape[3]
  except:
    # for pickle file from python2
    with open(label_path, 'rb') as f:
      sample_names, labels, sample_length = pickle.load(f)

  output = []
  label_output = []
  maxT = FLAGS.sub_seq_num ######

  def normalize_video(video):
    """Using 2*(x - min)/(max - min) - 1 normalization.
    :param video: np array of shape [seq_len, coordinate]
    :return:
    """
    max_75 = np.amax(video, axis=0)
    min_75 = np.amin(video, axis=0)
    max_x = np.max([max_75[i] for i in range(0,FLAGS.length_feature,3)])
    max_y = np.max([max_75[i] for i in range(1,FLAGS.length_feature,3)])
    max_z = np.max([max_75[i] for i in range(2,FLAGS.length_feature,3)])
    min_x = np.min([min_75[i] for i in range(0,FLAGS.length_feature,3)])
    min_y = np.min([min_75[i] for i in range(1,FLAGS.length_feature,3)])
    min_z = np.min([min_75[i] for i in range(2,FLAGS.length_feature,3)])
    norm = np.zeros_like(video)
    for i in range(0,FLAGS.length_feature,3):
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


  if data.shape[-1] == 60:  ## for N-UCLA data, (N, 75, 60)
    for seq, label in tqdm(zip(data, labels)):
      output.append(seq)
      label_output.append(label)

  elif data.shape[-1] == 45:  ## for UWA3D data, (N, 75, 45)
    for seq, label in tqdm(zip(data, labels)):
      output.append(seq)
      label_output.append(label)

  elif data.shape[-2] == 15:  ## for SBU data, (N, 2, 25, 15, 3)
    if FLAGS.length_feature == 75:
      ## Change to 25 joints, repeating with nearest joints
      hip = (data[:, :, :, 9:10, :] + data[:, :, :, 12:13, :]) / 2.0
      data = np.concatenate([data, hip], axis=3) # (N, 2, 25, 15 + 1, 3)
      data_new = np.zeros((data.shape[0], 2, 25, 25, 3))
      data_new[:, :, :, ntu_to_sbu, :] = data
      data_new[:, :, :, sbu_to_ntu[0], :] = data[:, :, :, sbu_to_ntu[1], :]
      # print (data_new.shape)
      data = data_new
    elif FLAGS.length_feature == 48: ## appending 'hip' keypoint
      hip = (data[:, :, :, 9:10, :] + data[:, :, :, 12:13, :]) / 2.0
      data = np.concatenate([data, hip], axis=3) # (N, 2, 25, 15 + 1, 3)
    else:
      raise Exception("SBU dataset not support this feature length: ", FLAGS.length_feature)

    N, M, T, V, C = data.shape
    for seq, label in tqdm(zip(data, labels)):
      seq = seq.reshape(M, T, -1)
      seq[0], seq[1] = normalize_video(seq[0]), normalize_video(seq[1])
      output.append(np.concatenate([seq[0], seq[0]]))
      label_output.append(label)

  else:  ## for NTU
    N, C, T, V, M = data.shape
    for seq, label, frame_num in tqdm(zip(data, labels, sample_length)):
      seq = seq.transpose((3,1,2,0)).reshape((M, T, -1))[0]  # only use first person
      # normalize first then downsample
      seq = normalize_video(seq)
      if frame_num <= maxT:
          seq = seq[:maxT]
      else:
          ## sample 'self.maxT' frames
          s = frame_num // maxT
          seq = seq[::s][:maxT]

      ## sampling points as SBU format
      if FLAGS.length_feature == 48:
        seq = np.reshape(seq, (maxT, V, C))[:, ntu_to_sbu, :].reshape(maxT, FLAGS.length_feature)

      output.append(seq)
      label_output.append(label)

  return output, label_output


def generate_masked_sample(input_, mode):
  output_ = [e for e in input_.copy()]
  masked_lm_position = []
  masked_lm_ids = []
  masked_lm_weights = []
  input_mask = [1 for _ in input_]
  segment_ids = [0 for _ in range(len(input_)//2)] + [1 for _ in range(len(input_)//2)]

  if 'finetune' not in mode and len(input_)!=0:
    factor = FLAGS.mask_length
    select_num = int(len(input_)*FLAGS.masked_lm_prob)//factor if int(len(input_)*FLAGS.masked_lm_prob)//factor!=0 else 1

    try:
      masked_index = random.sample(range( (len(input_)) // factor), select_num)
    except:
      print(range((len(input_))//factor), select_num)

    for i in masked_index:
      masked_lm_position += list(range(factor*i, factor*(i+1)))

    # output_: list of [2*T, 75]
    motion = np.zeros((FLAGS.sub_seq_num, FLAGS.length_feature))
    for t in range(1, FLAGS.sub_seq_num):
      motion[t - 1] = output_[t] - output_[t - 1]

    """Motion direction prediction data generation.
    """
    for pos in masked_lm_position:
      flow = motion[pos].reshape(FLAGS.length_feature//3, 3)
      x, y, z = flow[:, 0] > 0, flow[:, 1] > 0, flow[:, 2] > 0
      label = 1 * x + 2 * y + 4 * z
      offset = np.arange(FLAGS.length_feature//3) * 8 # every joint has 8 class
      # print(label)
      masked_lm_ids.append(label + offset)

    """Skeleton inpainting data generation.
    """
    # for ele in masked_lm_position:
    #   masked_frame = None
    #   if random.random() < 0.8: # 80% of the time, replace with
    #     masked_frame = np.zeros(FLAGS.length_feature, dtype=np.float32)
    #   else:
    #     if random.random() < 0.5: # 10% of the time, keep original
    #       masked_frame = output_[ele]
    #     else: # 10% of the time, replace with random word
    #       masked_frame = output_[random.randint(0, len(output_) - 1)]

    #   masked_lm_ids.append(output_[ele])
    #   output_[ele] = masked_frame

    masked_lm_weights = [1 for _ in masked_lm_ids]

  return output_, segment_ids, masked_lm_position, input_mask, masked_lm_ids, masked_lm_weights


def add_CLS_SEP(sub_seq_, masked_lm_position, input_mask):
  """Add [CLS] and [SEP] tokens into sequence,
     e.g. { [CLS], s_10, s_11, ..., s_1T, [SEP], s_20, s_21, ..., s_2T, [SEP] }
     where 'T' is the length of each person.
  """
  length_init = len(sub_seq_)
  sub_seq_.insert(0, -1.0*np.ones(FLAGS.length_feature, dtype = np.float32))
  sub_seq_.insert(length_init//2+1, 1.0*np.ones(FLAGS.length_feature, dtype = np.float32))
  sub_seq_.append(np.ones(FLAGS.length_feature, dtype = np.float32))

  for i in range(len(masked_lm_position)):
  	if masked_lm_position[i] >= length_init//2:
  		masked_lm_position[i] += 1
  masked_lm_position = [e+1 for e in masked_lm_position]

  input_mask += [1, 1, 1]

  return sub_seq_, masked_lm_position, input_mask


def padding(data):
  #[sub_seq_, input_mask, segment_ids, masked_lm_position, masked_lm_ids, masked_lm_weights, action_labels]
  #     0          1           2                    3               4                5            6
  data[0] += [np.zeros(FLAGS.length_feature, dtype=np.float32) for _ in range(FLAGS.max_seq_length-len(data[0]))]
  data[1] += [0 for _ in range(FLAGS.max_seq_length-len(data[1]))]
  data[2] += [0 for _ in range(FLAGS.max_seq_length-len(data[2]))]
  data[3] += [0 for _ in range(FLAGS.max_predictions_per_seq-len(data[3]))]
  data[4] += [np.zeros((FLAGS.length_feature // 3), dtype=np.int64) for _ in range(FLAGS.max_predictions_per_seq-len(data[4]))]
  data[5] += [0 for _ in range(FLAGS.max_predictions_per_seq-len(data[5]))]

  return data


def save_to_tfrecorder(data, writer):

  def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  #[sub_seq_, input_mask, segment_ids, masked_lm_position, masked_lm_ids, masked_lm_weights, action_labels]
  #     0          1           2                    3               4                5            6
  data[0] = np.stack(data[0]).astype(np.float32).reshape((-1))
  data[1] = np.stack(data[1]).astype(np.int64)
  data[2] = np.stack(data[2]).astype(np.int64)
  data[3] = np.stack(data[3]).astype(np.int64)

  if data[4][0].dtype == np.int or data[4][0].dtype == int:
    # for motion, it's classification, so labels are 'Int'
    data[4] = np.stack(data[4]).astype(np.int64).reshape((-1))
  else:
    # for inpainting, it's re-construction, so labels are 'float' (3D coordinate)
    data[4] = np.stack(data[4]).astype(np.float32).reshape((-1))

  data[5] = np.stack(data[5]).astype(np.float32)

  ## the shape of the input_mask and masked_lm_ids is [number_of_samples, 42]
  feature_dict = {
      'input_ids': float_feature(data[0]),
      'input_mask': int64_feature(data[1]),
      'segment_ids': int64_feature(data[2]),
      'masked_lm_positions': int64_feature(data[3]),
      'masked_lm_ids': int64_feature(data[4]) if data[4][0].dtype == np.int or data[4][0].dtype == int else float_feature(data[4]) ,
      'masked_lm_weights': float_feature(data[5]),
      'action_labels': int64_feature([data[6]]),
      }

  tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  writer.write(tf_example.SerializeToString())
  return writer

def create_data_for_tfrecorder(seqs, labels, mode):
  """Arrange .tfrecord file according to 'mode'
     'train': tf_train.tfrecord
     'eval':  tf_valid.tfrecord
     'finetune_train': tf_finetune_train.tfrecord
     'finetune_eval':  tf_finetune_eval.tfrecord
  """

  if seqs is None or labels is None:
    return

  if mode == 'train':
    name = FLAGS.output_file_train
  elif mode == 'eval':
    name = FLAGS.output_file_valid
  elif 'finetune' in mode:
    name = FLAGS.output_file_finetune
    name = name.replace("tf_finetune.tfrecord", "tf_{}.tfrecord".format(mode))

  print (name)
  writer = tf.python_io.TFRecordWriter(name)

  count = 0

  for idx, seq in tqdm(enumerate(seqs)):
    label = labels[idx]

    if 'finetune' in mode:
      repeat_num = 1
    else:
      repeat_num = 2

    for repeat_idx in range(repeat_num):
      sub_seq_ = seq.copy()
      sub_seq_, segment_ids, masked_lm_position, input_mask, masked_lm_ids, masked_lm_weights = \
          generate_masked_sample(sub_seq_, mode)

      # group
      data_to_write = [sub_seq_, input_mask, segment_ids, masked_lm_position,
                       masked_lm_ids, masked_lm_weights, label]
      # padding
      data_to_write = padding(data_to_write)

      # ## save data to tf_recoder
      writer = save_to_tfrecorder(data_to_write, writer)
      count += 1

      if count%1000 == 0:
        print('process {} data'.format(count))

  print('Totally prepare {} sequences'.format(count))
  writer.close()

  return

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.logging.info("*** Reading from input files ***")

  train_seqs, labels_train = None, None
  if os.path.exists(os.path.join(FLAGS.data_dir, 'train_data_{}.npy'.format(FLAGS.data_struc))):
    train_seqs, labels_train = read_data(FLAGS.data_dir, 'train', FLAGS.data_struc)

  valid_seqs, labels_valid = None, None
  if os.path.exists(os.path.join(FLAGS.data_dir, 'val_data_{}.npy'.format(FLAGS.data_struc))):
    valid_seqs, labels_valid = read_data(FLAGS.data_dir, 'val', FLAGS.data_struc)


  tf.logging.info("*** Writing training data to output files for train ***")
  create_data_for_tfrecorder(train_seqs, labels_train, mode='train')

  tf.logging.info("*** Writing no masked data to output files for finetune ***")
  create_data_for_tfrecorder(train_seqs, labels_train, mode='finetune_train')
  tf.logging.info("*** Writing no masked data to output files for finetune evaluation***")
  create_data_for_tfrecorder(valid_seqs, labels_valid, mode='finetune_valid')

if __name__ == "__main__":
  tf.app.run()

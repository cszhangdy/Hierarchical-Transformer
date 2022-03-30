# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import optimization
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 5,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "freeze_vars", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "use_seq_mean", False,
    "Whether to add l2 regularization to final loss.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer("length_feature", 99, "Dimension of the 3d pose vector.")

flags.DEFINE_integer("num_classes", 60, "")

flags.DEFINE_integer("num_joint_units", 16, "Dimension of each joint in decoder's transformation.")  #

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("is_finetune", False, "")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_string("model", None, "")

# modeling_action: no hierarchical
# modeling_hierarchical: hierarchical fusion process
if FLAGS.model == 'modeling_action':
  import modeling_action as modeling
elif FLAGS.model == 'modeling_hierarchical':
  import modeling_hierarchical as modeling
else:
  raise ValueError("Modeling type error: %s" % (FLAGS.model))


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    action_labels = features["action_labels"]


    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    ## Get attention maps from model
    attention_maps = model.get_attention_maps()
    embedding_last_layer = model.get_sequence_output()


    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_predictions) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    if FLAGS.use_seq_mean:
      # use sequence_output's mean for classification
      (classification_loss, classification_example_loss,
      classification_log_probs) = get_action_classification_output(
          bert_config, tf.reduce_mean(model.get_sequence_output(), axis=1), action_labels)
    else:
      (classification_loss, classification_example_loss,
      classification_log_probs) = get_action_classification_output(bert_config, mode.get_pooled_output(), action_labels)

    # Setting this loss because of TF's static graph building.
    # First, we run pre-training step. After that, we run finetuning step.
    # If we don't run classification in pre-training, i.e. no counting `cls` loss into final loss,
    # those variables seem not to be created and initialized (for what I remembered for this bug).
    # This causes in finetuning step, the pre-trained model contains no `cls` variables,
    # and in addition, the fxxking static graph won't allow us to append new variables in an easy way.
    # So Both pre-training and finetuning steps should be run.
    if FLAGS.is_finetune:
      total_loss = 1.0*classification_loss + 0.0*masked_lm_loss
    else:
      total_loss = 0.0*classification_loss + 1.0*masked_lm_loss

    # Initialize variables from checkpoint file, mostly for loading pre-trained model.
    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #   init_string = ""
    #   if var.name in initialized_variable_names:
    #     init_string = ", *INIT_FROM_CKPT*"
    #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                   init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_vars_list = None
      if FLAGS.freeze_vars:
        ## Freeze paramters except the following ones:
        train_vars_list = ['action_recognition']

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, train_vars_list)

      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=100)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          # total loss
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_predictions, masked_lm_ids, classification_log_probs, action_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        # masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=tf.ones_like(masked_lm_example_loss))
            # values=masked_lm_example_loss, weights=masked_lm_weights)

        masked_lm_predictions = tf.reshape(masked_lm_predictions, [-1, masked_lm_predictions.shape[-1]])
        flow_predictions = tf.argmax(masked_lm_predictions, axis=-1, output_type=tf.int32)
        flow_labels = tf.reshape(masked_lm_ids, [-1])
        flow_accuracy = tf.metrics.accuracy(labels=flow_labels, predictions=flow_predictions)


        classification_log_probs = tf.reshape(
            classification_log_probs, [-1, classification_log_probs.shape[-1]])
        classification_predictions = tf.argmax(
            classification_log_probs, axis=-1, output_type=tf.int32)

        action_labels = tf.reshape(action_labels, [-1])
        action_accuracy = tf.metrics.accuracy(
            labels=action_labels, predictions=classification_predictions)
        classification_mean_loss = tf.metrics.mean(
            values=classification_example_loss)

        return {
            "masked_lm_acc--flow pred": flow_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "action_accuracy": action_accuracy,
            "classification_mean_loss": classification_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_predictions, masked_lm_ids, classification_log_probs, action_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          # total loss
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)


    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
            'masked_lm_predictions': masked_lm_predictions,
            'classification_log_probs': classification_log_probs,
            'input_ids': input_ids,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'action_labels': action_labels,
            'attention_maps': attention_maps,
            'embedding_last_layer': embedding_last_layer,
          },
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM.
     Modified for flow prediction as a classification task
  """
  input_tensor = gather_indexes(input_tensor, positions) ## (N, length_feature)

  with tf.variable_scope("motion/predictions"):
    ## We apply one more non-linear transformation before the output layer.
    ## This matrix is not used after pre-training.
    num_joints = FLAGS.length_feature // 3  # NTU: 25, N-UCLA: 20
    with tf.variable_scope("body_transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=num_joints * FLAGS.num_joint_units,  # 0806h, hidden_size==2048, units=num_joints * 64
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    input_tensor = tf.reshape(input_tensor, [-1, num_joints, FLAGS.num_joint_units])  # [bs, num_joint, 12]

    joint_transform_unit = FLAGS.num_joint_units * 4
    with tf.variable_scope("joint_transform"):
      input_tensor = tf.layers.dense(
        input_tensor,
        units=joint_transform_unit,  # unit size for flow predict
        activation=modeling.get_activation(bert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
            bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    output_weights = tf.get_variable(
        "flow_cls_weights",
        shape=[num_joints*8, joint_transform_unit],
        initializer=modeling.create_initializer(bert_config.initializer_range)) # stddev=0.02 maybe too small for cls
    output_bias = tf.get_variable(
        "flow_cls_bias", shape=[num_joints*8], initializer=tf.zeros_initializer())

    input_tensor = tf.reshape(input_tensor, [-1, joint_transform_unit])
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True) ## (N, 25, 200), every position should predict 25 joints' flow, each joint has 8 direction
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # print (label_ids.shape, log_probs.shape)
    # input()
    label_ids = tf.reshape(label_ids, [-1])
    one_hot_labels = tf.one_hot(label_ids, depth=num_joints*8, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

  return (loss, per_example_loss, logits)


def get_action_classification_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # C-class classification for action recognition.
  # This weight matrix is only trained in finetuning step with labeled data.
  with tf.variable_scope("cls/action_recognition"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[FLAGS.num_classes, int(input_tensor.shape[-1])],
        # initializer=modeling.create_initializer(math.sqrt(2. / FLAGS.num_classes)))
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[FLAGS.num_classes], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)



def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])

  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length, FLAGS.length_feature], tf.float32),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq, FLAGS.length_feature//3], tf.int64), ### for cls, tf.int64
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "action_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.

      ################# to avoid endless output, comment this line #################
      # d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # Get `run_config` for model running
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      session_config=config)

  # Get `model_fn`
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)
  tf.logging.info("***** model_fn created *****")

  # Get estimator, with `model_fn` and `run_config`
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size) ## add predict
  tf.logging.info("***** estimator created *****")

  # TRAIN: get `train_input_fn`, for generating data from .tfrecord file.
  # Then send data to estimator.
  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  # EVAL: get `eval_input_fn`, for generating data from .tfrecord file.
  # Then send data to estimator.
  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn) #, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  # PREDICT: get `predict_input_fn`, for generating data from .tfrecord file.
  # Then send data to estimator. Results are saved in `predict_results`, where
  # every batch you get a dict type `result`. See following for details.
  if FLAGS.do_predict:

    tf.logging.info("***** Running prediction *****")
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    predict_results = estimator.predict(
        input_fn=predict_input_fn, yield_single_examples=False)

    start_index = 0; end_index = 1000
    result = {}
    result['attention_maps'] = []
    result['labels'] = []

    for i, p in tqdm(enumerate(predict_results)):
        result['labels'].append(p['action_labels'])
        result['attention_maps'].append(p['attention_maps']) # (B, N, F, T)
        # attentions = np.split(p['attention_maps'], [5, 10, 14, 18, 20, 22, 23], axis=1)
        # # print (p['attention_maps'].shape, p['action_labels'].shape)
        # for j, attn in enumerate(attentions):
        #   attention_maps[j].append(attn)
        # input()
        if i >= end_index:
          break


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

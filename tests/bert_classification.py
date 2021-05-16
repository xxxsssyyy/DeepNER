# encoding=utf8
import os
import codecs
import pickle
import itertools
import sys
import math
import random
from collections import OrderedDict
sys.path.append('../')

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from bert import modeling, tokenization


class BertBatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        #sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(data[int(i*batch_size) : int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def arrange_batch(batch):
        '''
        batch as a [3, ] array
        :param batch:
        :return:
        '''
        word_id_list = []
        word_mask_list = []
        word_segment_list = []
        label_list = []

        for word_ids, input_masks, word_segment_ids, label in batch:
            word_id_list.append(word_ids)
            word_mask_list.append(input_masks)
            word_segment_list.append(word_segment_ids)
            label_list.append(label)
        return [word_id_list, word_mask_list, word_segment_list, label_list]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class BertClassification(object):
    def __init__(self, config):
        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.max_seq_lens = config['max_seq_lens']
        self.batch_size = config['batch_size']

        self.pooling_strategy = config['pooling_strategy']
        self.pooling_layer = config['pooling_layer']
        self.init_checkpoint = config['init_checkpoint']
        self.config_file = config['config_file']
        self.vocab_file = config['vocab_file']
        self.freeze_bert = config['freeze_bert']
        self.num_tags = config['num_tags']
        self.lr = config['lr']
        self.lower = config['lower']
        self.epochs = config['epochs']
        self.steps_check = config['steps_check']
        self.save_path = config['save_path']
        self.train_file = config['train_file']
        self.dev_file = config['dev_file']
        self.test_file = config['test_file']
        self.save_test_result_file = config['save_test_result_file']

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_tags], name="labels")


        bert_config = modeling.BertConfig.from_json_file(self.config_file)
        
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        
        minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
        masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
        masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
        # load bert embedding
        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        with tf.variable_scope("pooling"):
            if len(self.pooling_layer) == 1:
                encoder_layer = model.all_encoder_layers[self.pooling_layer[0]]
            else:
                all_layers = [model.all_encoder_layers[l] for l in self.pooling_layer]
                encoder_layer = tf.concat(all_layers, -1)
            input_mask = tf.cast(self.input_mask, tf.float32)
            if self.pooling_strategy == 'REDUCE_MEAN':
                pooled = masked_reduce_mean(encoder_layer, input_mask)
            elif self.pooling_strategy == 'REDUCE_MAX':
                pooled = masked_reduce_max(encoder_layer, input_mask)
            elif self.pooling_strategy == 'REDUCE_MEAN_MAX':
                pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                    masked_reduce_max(encoder_layer, input_mask)], axis=1)
            elif self.pooling_strategy == 'FIRST_TOKEN' or \
                    self.pooling_strategy == 'CLS_TOKEN':
                pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
            elif self.pooling_strategy == 'CLS_POOLED':
                pooled = model.pooled_output
            elif self.pooling_strategy == 'LAST_TOKEN' or \
                    self.pooling_strategy == 'SEP_TOKEN':
                seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                rng = tf.range(0, tf.shape(seq_len)[0])
                indexes = tf.stack([rng, seq_len - 1], 1)
                pooled = tf.gather_nd(encoder_layer, indexes)

        #self.pooled = tf.identity(pooled, 'final_encodes') # [batch_size, 768]
        self.pooled = pooled
        #self.pooled = model.get_pooled_output()

        self.logits = tf.layers.dense(self.pooled, self.num_tags, name="dense_to_labels") # [batch_size, 2]
        self.norm_logits = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        #print(self.predictions.shape)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # print variables
        train_vars = []
        for var in tvars:
            init_string = ""
            # freeze bert parameters, only train the parameters of add networks
            if var.name in initialized_variable_names:
                if not self.freeze_bert:
                    train_vars.append(var)
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            print("  name = %s, shape = %s%s", var.name, var.shape,
                  init_string)

        with tf.variable_scope("optimizer"):
            
            self.opt = tf.train.AdamOptimizer(self.lr)

            grads = tf.gradients(self.loss, train_vars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = self.opt.apply_gradients(
                zip(grads, train_vars), global_step=self.global_step)
            #capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                     for g, v in grads_vars if g is not None]
            #self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step, )

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def create_feed_dict(self, is_train, batch):
        """
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        word_ids, input_masks, word_segment_ids, labels = batch
        feed_dict = {
            self.input_ids: np.asarray(word_ids),
            self.input_mask: np.asarray(input_masks),
            self.segment_ids: np.asarray(word_segment_ids)
        }
        if is_train:
            feed_dict[self.labels] = np.asarray(labels)
        return feed_dict
    
    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            logits = sess.run([self.logits], feed_dict)
            return logits

    def evaluate(self, sess, data_manager):
        predict_list = []
        true_list = []

        for batch in data_manager.iter_batch(shuffle=False):
            logits = self.run_step(sess, False, batch)
            predict_labels = np.argmax(logits[0], axis=1)
            predict_list.extend(predict_labels.tolist())
            for label in batch[3]:
                true_list.append(0 if label[0]==1 else 1)
        pre = precision_score(true_list, predict_list)
        recall = recall_score(true_list, predict_list)
        f1_score_1 = f1_score(true_list, predict_list)
        return pre, recall, f1_score_1

    def save_model(self, sess, path):
        checkpoint_path = os.path.join(path, "implicit_region_classification.ckpt")
        self.saver.save(sess, checkpoint_path)
        print("model saved")
    

    def read_input(self, filename):
        """
        # TODO: add lower function
        """
        token = tokenization.FullTokenizer(vocab_file=self.vocab_file)
        sentences = []
        for i, line in enumerate(codecs.open(filename, 'r', 'utf8')):
            # if i == 0:
            #     continue
            items = line.strip().split('\t')
            if len(items) != 2:
                continue
            label = int(items[0])
            sentence = items[1]
            sentences.append([sentence, label])
            
        data = []
        
        for sentence in sentences:
            split_tokens = token.tokenize(sentence[0])
            input_masks = [1] * len(split_tokens)
            
            # cut and pad sentence to max_seq_lens-2
            if len(split_tokens) > self.max_seq_lens-2:
                split_tokens = split_tokens[:self.max_seq_lens-2]
                split_tokens.append("[SEP]")
                input_masks = input_masks[:self.max_seq_lens-2]
                input_masks = [1] + input_masks + [1]
            else:
                split_tokens.append("[SEP]")
                input_masks.append(1)
                while len(split_tokens) < self.max_seq_lens-1:
                    split_tokens.append('[PAD]')
                    input_masks.append(0)
                input_masks.append(0)
            # add CLS and SEP for tokens
            tokens = []
            tokens.append("[CLS]")
            for i_token in split_tokens:
                tokens.append(i_token)
            
            word_ids = token.convert_tokens_to_ids(tokens)
            word_segment_ids = [0] * len(word_ids)

            label = sentence[1]
            label = [1, 0] if label == 0 else [0, 1]
            data.append([word_ids, input_masks, word_segment_ids, label])
        return data, sentences

    def train(self):
        train_data, train_sentences = self.read_input(self.train_file)
        dev_data, dev_sentences = self.read_input(self.dev_file)
        test_data, test_sentences = self.read_input(self.test_file)

        print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

        train_manager = BertBatchManager(train_data, self.batch_size)
        dev_manager = BertBatchManager(dev_data, self.batch_size)
        test_manager = BertBatchManager(test_data, self.batch_size)

        # limit GPU memory
        tf_config = tf.ConfigProto()
        #tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data

        with tf.Session(config=tf_config) as sess:
            
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

            ckpt = tf.train.get_checkpoint_state(self.save_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                #saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
                #saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
            
            print("start training")
            loss = []
            dev_best = 0.0
            for i in range(self.epochs):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = self.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % self.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        print("iteration:{} step:{}/{}, "
                                "loss:{:>9.6f}".format(
                            iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

                dev_pre, dev_recall, dev_f1 = self.evaluate(sess, dev_manager)
                if (dev_f1 > dev_best):
                    dev_best = dev_f1
                    self.save_model(sess, self.save_path)
                test_pre, test_recall, test_f1 = self.evaluate(sess, test_manager)
                print("iteration:{}, dev precision: {}, dev recall: {}, dev f1: {}, "
                                "test precision: {}, test recall: {}, test f1: {}".format(
                            iteration, dev_pre, dev_recall, dev_f1, test_pre, test_recall, test_f1))

            # predict test_data and write into file
            print("start predicting and saving test result")
            self.predict_and_save(sess, test_sentences, test_manager)
            print("finish training and predicting")

    def predict_and_save(self, sess, sentences, data_manager):
        predict_list = []
        true_list = []

        for batch in data_manager.iter_batch(shuffle=False):
            logits = self.run_step(sess, False, batch)
            predict_labels = np.argmax(logits[0], axis=1)
            predict_list.extend(predict_labels.tolist())
            for label in batch[3]:
                true_list.append(0 if label[0]==1 else 1)

        assert len(predict_list) == len(sentences)
        with codecs.open(self.save_test_result_file, 'w', 'utf8') as f:
            f.write("query\ttrue_label\tpredict_label\n")
            for i in range(len(predict_list)):
                f.write("%s\t%s\t%s\n" % (sentences[i][0], sentences[i][1], predict_list[i]))



if __name__ == "__main__":
    config = {
        'init_checkpoint': "../data/chinese_L-12_H-768_A-12/bert_model.ckpt",
        'config_file': "../data/chinese_L-12_H-768_A-12/bert_config.json",
        'vocab_file': "../data/chinese_L-12_H-768_A-12/vocab.txt",
        'pooling_strategy': 'CLS_TOKEN',
        'pooling_layer': [-1],
        'max_seq_lens': 128,
        'batch_size': 128,
        'freeze_bert': True,
        'num_tags': 2,
        'lr': 0.001,
        'lower': True,
        'epochs': 100,
        'steps_check': 20,
        'save_path': 'ckpt',
        'train_file': './exp1_data/data.train',
        'dev_file': './exp1_data/data.dev',
        'test_file': './exp1_data/data.test',
        'save_test_result_file': './test_result.txt'
    }
    model = BertClassification(config)
    model.train()
    
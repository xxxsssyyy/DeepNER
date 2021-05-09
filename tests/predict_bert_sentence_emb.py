# encoding=utf8
import os
import codecs
import pickle
import itertools
import sys
import math
from collections import OrderedDict
sys.path.append('../')

import tensorflow as tf
import numpy as np
from bert import modeling, tokenization

config = {
    'init_checkpoint': "../data/chinese_L-12_H-768_A-12/bert_model.ckpt",
    'config_file': "../data/chinese_L-12_H-768_A-12/bert_config.json",
    'vocab_file': "../data/chinese_L-12_H-768_A-12/vocab.txt",
    'pooling_strategy': 'CLS_TOKEN',
    'pooling_layer': [-1],
    'max_seq_lens': 128,
    'batch_size': 128
}


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

        for word_ids, input_masks, word_segment_ids in batch:
            word_id_list.append(word_ids)
            word_mask_list.append(input_masks)
            word_segment_list.append(word_segment_ids)
        return [word_id_list, word_mask_list, word_segment_list]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class BertVector:
    def __init__(self, config):
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        
        self.max_seq_lens = config['max_seq_lens']
        self.batch_size = config['batch_size']

        self.pooling_strategy = config['pooling_strategy']
        self.pooling_layer = config['pooling_layer']
        self.init_checkpoint = config['init_checkpoint']
        self.config_file = config['config_file']
        self.vocab_file = config['vocab_file']

        self.token = tokenization.FullTokenizer(vocab_file=self.vocab_file)
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

        self.pooled = tf.identity(pooled, 'final_encodes')

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
    
    def create_feed_dict(self, batch):
        """
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        word_ids, input_masks, word_segment_ids = batch
        feed_dict = {
            self.input_ids: np.asarray(word_ids),
            self.input_mask: np.asarray(input_masks),
            self.segment_ids: np.asarray(word_segment_ids),
        }
        return feed_dict
    
    def run_step(self, sess, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(batch)
        
        pooled = sess.run([self.pooled], feed_dict)
        return pooled

    def encode(self, sentences):
        if len(sentences) <= 0:
            raise ValueError()
        data = self.read_input(sentences)
        data_manager = BertBatchManager(data, self.batch_size)
        embs_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for batch in data_manager.iter_batch(shuffle=False):
                batch_embs = self.run_step(sess, batch)
                
                embs_list.append(batch_embs[0])
                
            embeddings = np.vstack(embs_list)
        return embeddings

    def read_input(self, sentences):
        data = []
        max_len = max([len(single) for single in sentences]) 
        
        for sentence in sentences:
            split_tokens = self.token.tokenize(sentence)
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
            
            word_ids = self.token.convert_tokens_to_ids(tokens)
            word_segment_ids = [0] * len(word_ids)

            data.append([word_ids, input_masks, word_segment_ids])
        return data

    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    bc = BertVector(config)
    embs = bc.encode(["腾讯微信搜索应用部。", "中国科学院自动化研究所。", "这里可真不错", "今天天气不错，适合出行。", "今天是晴天，可以出去玩。"])

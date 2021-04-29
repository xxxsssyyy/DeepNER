# encoding = utf8
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils.utils import result_to_json
from utils.data_utils import create_input, iobes_iob
from layers.attention import positional_encoding
from layers.transformer import Encoder
from layers.modules import *

class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]
        self.layers = config["layers"]
        self.heads = config["heads"]
        self.max_seq_len = config["max_seq_len"]
        self.num_blocks = config["num_blocks"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, self.max_seq_len],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, self.max_seq_len],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, self.max_seq_len],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
        
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        """
        # embeddings for chinese character and segmentation representation and positional embeddings
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        # mask for padding
        self.input_masks = tf.math.equal(self.char_inputs, 0)
        # attention_mask = self.create_attention_mask_from_input_mask(self.char_inputs, self.input_mask)
        # apply dropout before feed to transformer layer
        transformer_inputs = tf.nn.dropout(embedding, self.dropout) 
        # bi-directional transformer layer
        self.encoder = Encoder(num_layers=self.layers, num_heads=self.heads, linear_key_dim=self.lstm_dim, linear_value_dim=self.lstm_dim, 
                model_dim=self.lstm_dim, ffn_dim=self.lstm_dim, dropout=self.dropout)
        #lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)
        transformer_outputs = self.encoder.build(transformer_inputs, self.input_masks)
        print('transformer_outputs shape: ', transformer_outputs.shape)
        """
        embs = embedding(self.char_inputs, vocab_size=self.num_chars, num_units=self.char_dim, scale=True, scope="embed")
        transformer_outputs = self.encoder(embs)

        # logits for tags [Batch_size, num_steps, num_tags]
        self.logits = self.project_layer(transformer_outputs)
        self.max_logits = tf.argmax(self.logits, -1)
        print('logits shape: ', self.logits.shape)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """
        
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name): # tf.device('/cpu:0')
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            self.char_lookup = tf.concat((tf.zeros(shape=[1, self.char_dim]), self.char_lookup[1:, :]), 0)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"): # tf.device('/cpu:0')
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            
            embed = tf.concat(embedding, axis=-1)
            print('embed shape: ', embed.shape)
            # Positional Encoding
            with tf.variable_scope("positional-encoding"):
                #print('1: %s, 2: %s' % (embed.get_shape().as_list()[-1], self.char_inputs.get_shape().as_list()[-1]))
                positional_encoded = positional_encoding(embed.get_shape().as_list()[-1],
                                                         self.max_seq_len,
                                                         dtype=tf.float32)
                
                position_inputs = tf.tile(tf.range(0, self.num_steps), [self.batch_size])
                position_inputs = tf.reshape(position_inputs,
                                            [self.batch_size, self.num_steps]) # batch_size x [0, 1, 2, ..., n]
                position_embed = tf.nn.embedding_lookup(positional_encoded, position_inputs)
                print('position_embed  shape: ', position_embed.shape)
            all_embed = tf.add(embed, position_embed) # Add
        return all_embed

    def encoder(self, embed):
        with tf.variable_scope("Transformer_Encoder"):
			# Positional Encoding
            embed += positional_encoding(self.char_inputs, num_units=embed.get_shape().as_list()[-1], zero_pad=False, scale=False, scope="enc_pe")
			# Dropout
			# embed = tf.layers.dropout(embed, rate=hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            output = self.multi_head_block(embed, embed)
            return output
    
    def multi_head_block(self, query, key, decoding=False, causality=False):
        """
		:param query:
		:param key:
		:param decoding:
		:param causality:
		:return:
		"""
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
			    # multi head Attention ( self-attention)
                query = multihead_attention(
					queries=query, keys=key, num_units=self.lstm_dim, num_heads=self.heads,
					dropout_rate=self.dropout, is_training=True, causality=causality,
					scope="self_attention")
                if decoding:
					# multi head Attention ( vanilla attention)
                    query = multihead_attention(
						queries=query, keys=key, num_units=self.lstm_dim, num_heads=self.heads,
						dropout_rate=self.dropout, is_training=True, causality=False,
						scope="vanilla_attention")
				# Feed Forward
                query = feedforward(query, num_units=[4 * self.lstm_dim, self.lstm_dim])
        return query

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size*2] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            w = tf.get_variable(name='w', dtype=tf.float32, shape=[self.lstm_dim, self.num_tags])
            b = tf.get_variable(name='b', dtype=tf.float32, shape=[self.num_tags])
            
            lstm_outputs = tf.reshape(lstm_outputs, [-1, self.lstm_dim])
            logits = tf.matmul(lstm_outputs, w) + b
            logits = tf.reshape(logits, [-1, self.max_seq_len, self.num_tags])
            return logits

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [batch_size, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            """
            small = -1000.0
            # pad logits for crf loss
            # add a extra label
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1) # [batch_size, 1, num_tags+1]
            # add small value for extra label in logits
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1) # [batch_size, num_steps, num_tags+1]
            logits = tf.concat([start_logits, logits], axis=1) # [batch_size, num_steps+1, num_tags+1]
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1) # [batch_size, num_tags+1]

            # transition scores
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            """
            log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                project_logits, self.targets, lengths
            )
            
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
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
            #temp_msk = sess.run([self.encoder.att.temp_masks], feed_dict)
            #print('temp_msk shape: ', temp_msk[0].shape)
            #h = temp_msk[0].reshape([-1, self.max_seq_len])
            #h = np.sum(h, -1)
            #print(temp_msk[0])
            """ see logits
            lg = sess.run([self.max_logits], feed_dict)
            print(lg[0].shape)
            exist = (lg[0] > 0) * 1.0
            factor = np.ones(lg[0].shape[1])
            res = np.dot(exist, factor)
            print(res)
            """
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        pre_seqs = []
        for score, seq_len in zip(logits, lengths):
            pre_seq, pre_score = viterbi_decode(score[:seq_len], matrix)
            pre_seqs.append(pre_seq)
        return pre_seqs

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        print(tags)
        return result_to_json(inputs[0][0], tags)
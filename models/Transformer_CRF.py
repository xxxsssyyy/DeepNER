# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import tensorflow as tf
from layers.modules import *


class TransformerCRFModel(object):
	def __init__(self, config, is_training=True):
		self.vocab_size = config["num_chars"]
		self.num_tags = config["num_tags"]
		self.is_training = is_training
		self.graph = tf.Graph()
		self.max_len = config["max_seq_len"]
		self.num_units = config["num_units"]
		self.dropout_rate = config["dropout"]
		self.num_blocks = config["num_blocks"]
		self.num_heads = config["heads"]
		self.lr = config["lr"]

		
		self.x = tf.placeholder(tf.int32, shape=(None, self.max_len))
		self.y = tf.placeholder(tf.int32, shape=(None, self.max_len))
		used = tf.sign(tf.abs(self.x))
		length = tf.reduce_sum(used, reduction_indices=1)
		self.seq_lens = tf.cast(length, tf.int32)
		#self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
		self.global_step = tf.train.create_global_step()
		
		# layers embedding multi_head_attention rnn
		outputs = embedding(self.x, vocab_size=self.vocab_size, num_units=self.num_units, scale=True, scope="embed")
		
		outputs = self.encoder(outputs)
		# outputs = self.cnn_layer(outputs)
		#outputs = self.rnn_layer(outputs)
		self.logits = self.logits_layer(outputs)
		self.loss, self.transition = self.crf_layer()
		self.train_op = self.optimize()
		
		# saver of the model
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
	
	def encoder(self, embed):
		with tf.variable_scope("Transformer_Encoder"):
			# Positional Encoding
			embed += positional_encoding(self.x, num_units=self.num_units, zero_pad=False, scale=False, scope="enc_pe")
			# Dropout
			embed = tf.layers.dropout(embed, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
			output = self.multi_head_block(embed, embed)
			return output
	
	def decoder(self, enc):
		"""
		:param enc:
		:return:
		"""
		with tf.variable_scope("Transformer_Decoder"):
			# Embedding
			dec = embedding(self.y, vocab_size=self.num_tags, num_units=self.num_units, scale=True, scope="dec_embed")
			
			# Positional Encoding
			dec += positional_encoding(self.y, num_units=self.num_units, zero_pad=False, scale=False, scope="dec_pe")
			# Dropout
			dec = tf.layers.dropout(dec, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
			
			output = self.multi_head_block(dec, enc, decoding=True, causality=True)
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
					queries=query, keys=key, num_units=self.num_units, num_heads=self.num_heads,
					dropout_rate=self.dropout_rate, is_training=self.is_training, causality=causality,
					scope="self_attention")
				if decoding:
					# multi head Attention ( vanilla attention)
					query = multihead_attention(
						queries=query, keys=key, num_units=self.num_units, num_heads=self.num_heads,
						dropout_rate=self.dropout_rate, is_training=self.is_training, causality=False,
						scope="vanilla_attention")
				# Feed Forward
				query = feedforward(query, num_units=[4 * self.num_units, self.num_units])
		return query
	
	def logits_layer(self, outputs):
		"""
		logits
		:param outputs:
		:return:
		"""
		w = tf.get_variable(name='w', dtype=tf.float32, shape=[self.num_units, self.num_tags])
		b = tf.get_variable(name='b', dtype=tf.float32, shape=[self.num_tags])
		
		outputs = tf.reshape(outputs, [-1, self.num_units])
		logits = tf.matmul(outputs, w) + b
		logits = tf.reshape(logits, [-1, self.max_len, self.num_tags])
		return logits
	
	def crf_layer(self):
		log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(self.logits, self.y, self.seq_lens)
		loss = tf.reduce_mean(-log_likelihood)
		return loss, transition
	
	def optimize(self):
		"""
		:return:
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		return train_op
	
	def predict(self, logits, transition, seq_lens):
		pre_seqs = []
		for score, seq_len in zip(logits, seq_lens):
			pre_seq, pre_score = tf.contrib.crf.viterbi_decode(score[:seq_len], transition)
			pre_seqs.append(pre_seq)
		return pre_seqs
	
	def create_feed_dict(self, is_train, batch):
		"""
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
		"""
		_, chars, segs, tags = batch
		feed_dict = {
            self.x: np.asarray(chars),
        }
		if is_train:
			feed_dict[self.y] = np.asarray(tags)
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
			lengths, logits = sess.run([self.seq_lens, self.logits], feed_dict)
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
			pre_seq, pre_score = tf.contrib.crf.viterbi_decode(score[:seq_len], matrix)
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
		trans = self.transition.eval()
		for batch in data_manager.iter_batch():
			strings = batch[0]
			tags = batch[-1]
			lengths, scores = self.run_step(sess, False, batch)
			batch_paths = self.decode(scores, lengths, trans)
			for i in range(len(strings)):
				result = []
				string = strings[i][:lengths[i]]
				gold = [id_to_tag[int(x)] for x in tags[i][:lengths[i]]]
				pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]
				for char, gold, pred in zip(string, gold, pred):
					result.append(" ".join([char, gold, pred]))
				results.append(result)
		return results
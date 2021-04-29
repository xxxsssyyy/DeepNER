# encoding = utf8
# this file is copy from: https://github.com/DongjunLee/transformer-tensorflow
import tensorflow as tf

import sys
sys.path.append('../')

from layers.attention import Attention


class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""
    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self):
        raise NotImplementedError("i will implement it!")


class Encoder:
    """Encoder class"""
    def __init__(self,
                 num_layers=8,
                 num_heads=8,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=50,
                 ffn_dim=50,
                 max_seq_len=100,
                 dropout=0.2,
                 ):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

    def build(self, encoder_inputs, key_masks):
        o1 = tf.identity(encoder_inputs) # reference passing

        for i in range(1, self.num_layers+1):
            with tf.variable_scope(f"layer-{i}"):
                o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1,
                                                                 key_masks=key_masks), num=1)
                o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                o1 = tf.identity(o3)

        return o3

    def _self_attention(self, q, k, v, key_masks):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=False,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    max_seq_len=self.max_seq_len,
                                    dropout=self.dropout,
                                    )
            #self.att = attention
            return attention.multi_head(q, k, v, key_masks)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope(f"encoder-add-and-norm-{num}"):
            # Layer Normalization with Residual connection
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x))

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)


class Decoder:
    """Decoder class"""
    def __init__(self,
                 num_layers=8,
                 num_heads=8,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=50,
                 ffn_dim=50,
                 dropout=0.2):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def build(self, decoder_inputs, encoder_outputs):
        o1 = tf.identity(decoder_inputs)

        for i in range(1, self.num_layers+1):
            with tf.variable_scope(f"layer-{i}"):
                o2 = self._add_and_norm(o1, self._masked_self_attention(q=o1,
                                                                        k=o1,
                                                                        v=o1), num=1)
                o3 = self._add_and_norm(o2, self._encoder_decoder_attention(q=o2,
                                                                            k=encoder_outputs,
                                                                            v=encoder_outputs), num=2)
                o4 = self._add_and_norm(o3, self._positional_feed_forward(o3), num=3)
                o1 = tf.identity(o4)

        return o4

    def _masked_self_attention(self, q, k, v):
        with tf.variable_scope("masked-self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,  # Not implemented yet
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope(f"decoder-add-and-norm-{num}"):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _encoder_decoder_attention(self, q, k, v):
        with tf.variable_scope("encoder-decoder-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=False,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)
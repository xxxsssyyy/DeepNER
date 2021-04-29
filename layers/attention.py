# encoding = utf8
# this file is copy from: https://github.com/DongjunLee/transformer-tensorflow
import numpy as np
import tensorflow as tf

__all__ = [
    "positional_encoding", "Attention"
]


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    # 2i: sin(\frac{pos}{10000^{2*i/dim}}); 2i+1: cos(\frac{pos}{10000^{2*i/dim}})
    # i is the dimention of emb
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2]) # three parameters: start = 0, step = 2
    encoded_vec[1::2] = np.cos(encoded_vec[1::2]) # three parameters: start = 1, step = 2

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class Attention:
    """Attention class"""

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 max_seq_len=100,
                 dropout=0.2,
                 ):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

    def multi_head(self, q, k, v, key_masks):
        """
        :param q: [batch_size, max_seq_len, emb_size]
        """
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, key_masks)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim) # [batch_size, max_seq_len, model_dim]

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        # q, k: [batch_size, max_seq_len, linear_key_dim], v: [batch_size, max_seq_len, linear_value_dim]
        return q, k, v 

    def _split_heads(self, q, k, v):
        # divide embeddings to several chunks
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            #tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            tensor = tf.reshape(tensor, [-1] + [self.max_seq_len] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _mask(self, inputs, key_masks=None, type=None):
        # refer: https://github.com/Kyubyong/transformer/blob/fb023bb097e08d53baf25b46a9da490beba51a21/modules.py#L103
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            key_masks = tf.to_float(key_masks) # [batch_size, max_seq_len]
            key_masks = tf.tile(key_masks, [self.num_heads, 1]) # [batch_size*num_heads, max_seq_len]
            key_masks = tf.expand_dims(key_masks, 1)  # [batch_size*num_heads, 1, max_seq_len]
            key_masks = tf.reshape(key_masks, [-1, self.num_heads, 1, self.max_seq_len]) # [batch_size, num_heads, 1, max_seq_len]
            outputs = inputs + key_masks * padding_num # [batch_size, num_heads, max_seq_len, max_seq_len]
        else:
            print("Check if you entered type correctly!")
        return outputs

    def _scaled_dot_product(self, qs, ks, vs, key_masks):
        # $softmax(\frac{Q*K}{\sqrt{d}})*V$ 
        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5) # [batch_size, num_heads, max_seq_len, max_seq_len]

        # key masking
        o2 = self._mask(o2, key_masks=key_masks, type="key") # [batch_size, num_heads, max_seq_len, max_seq_len]

        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :]) # o2: [batch_size, num_heads, max_seq_len, max_seq_len] -> [max_seq_len, max_seq_len]
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # transform matrix to lower triangular matrix: [max_seq_len, max_seq_len]
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            # tf.tile([1, 1, max_seq_len, max_seq_len], [batch_size, num_heads, 1, 1]) -> [batch_size, num_heads, max_seq_len, max_seq_len]
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)
        #self.temp_masks = o3
        return tf.matmul(o3, vs) # [batch_size, num_heads, max_seq_len, linear_value_dim]

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim]) # [batch_size, max_seq_len, emb_dim]

        return transpose_then_concat_last_two_dimenstion(outputs)
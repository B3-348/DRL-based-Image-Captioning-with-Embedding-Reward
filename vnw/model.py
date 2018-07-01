import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import math

from tensorflow.python.framework import ops


def selu(x):
    """
        This function is to implement selu activation function
        Selu(x) = scale * x if x > 0 else scale * (alpha * x - alpha)
    Args:
        x: input data
    Returns:
        Selu(x)
    """
    with ops.name_scope('elu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.nn.elu(x))


def lecun_normal(dim_in, dim_out, name=None, stddev=1.0):
    return tf.get_variable(name=name,
                           initializer=tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))))


class ValueNetWork(object):
    def __init__(self, batch_size=100, vocab_size=10000, embed_size=512,
                 learning_rate=0.001, feature_o_dim=4096, feature_n_dim=512, hidden_units=512,
                 num_layers=1, max_time_step=17):
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.9)
        self.num_layers = num_layers

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.feature_o_dim = feature_o_dim
        self.feature_n_dim = feature_n_dim
        self.hidden_units = hidden_units
        self.mlp_layer1_dim = 1024
        self.mlp_layer2_dim = 512

        self.cell = tc.rnn.GRUCell

        # weight initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

        # variable to feat
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, 4096], name="image_features")
        self.captions = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sample_t_captions")
        self.captions_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name="sample_t_captions_length")
        self.original_caption = tf.placeholder(dtype=tf.int32, shape=[None, max_time_step], name="original_caption")

    def _batch_norm(self, x, mode="train", name=None):
        return tc.layers.batch_norm(inputs=x,
                                    decay=0.95,
                                    center=True,
                                    scale=True,
                                    is_training=mode,
                                    updates_collections=None,
                                    scope=(name+'batch_norm')
                                    )

    def _proj_features(self, features):
        """
            To project features of 4096 dims to 512 dims
        Args:
            features: shape:(batch_size, 4096)
        Returns:
            feature_proj: shape:(batch_size, 512)
        """
        with tf.variable_scope("proj_features"):
            w = tf.get_variable("w", shape=[self.feature_o_dim, self.feature_n_dim],
                                initializer=self.weight_initializer)
            b = tf.get_variable("b", shape=[self.feature_n_dim], initializer=self.const_initializer)

            feature_proj = tf.matmul(features, w) + b

            return feature_proj

    def train_op(self, loss, is_clip_gradient=False):
        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

            return train_op

    def _loss(self, _v, _r):
        """
            loss = ||v(st) - r||^2 where v is value network's prediction and r is the real sentence reward
        Args:
            _v shape:(batch_size, 1)
            _r shape:(batch_size, 1)
        Returns:
            loss
        """
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square((_v - _r)), axis=0))
            return loss

    def _mlp_layer(self, x, activation_function=tf.nn.relu):
        """
            To implement mlp layers of value network, where x is concatenation of feature_proj and rnn_output
        Args:
            x:concatenation of feature_proj and rnn_output, which has a shape of (batch_size, 1024)
            activation_function: activation_function used for hidden layers
        Returns:
            v:value network's prediction, and it's shape is (batch_size, 1)
        """
        with tf.variable_scope("mlp_layer"):
            # variable of three-layers MLP
            layer1_w = tf.get_variable("layer1_w", shape=[self.hidden_units+self.feature_n_dim, self.mlp_layer1_dim],
                                       initializer=self.weight_initializer)
            layer1_b = tf.get_variable("layer1_b", shape=[self.mlp_layer1_dim], initializer=self.const_initializer)

            layer2_w = tf.get_variable("layer2_w", shape=[self.mlp_layer1_dim, self.mlp_layer2_dim],
                                       initializer=self.weight_initializer)
            layer2_b = tf.get_variable("layer2_b", shape=[self.mlp_layer2_dim], initializer=self.const_initializer)

            layer3_w = tf.get_variable("layer3_w", shape=[self.mlp_layer2_dim, 1], initializer=self.weight_initializer)

            layer3_b = tf.get_variable("layer3_b", shape=[1], initializer=self.const_initializer)

            # Layer1
            layer1_output = activation_function(tf.matmul(x, layer1_w) + layer1_b)  # shape: [batch_size, 1024]

            # Layer2
            # shape: [batch_size, 512]
            layer2_output = activation_function(tf.matmul(layer1_output, layer2_w) + layer2_b)

            # Layer3
            v = tf.matmul(layer2_output, layer3_w) + layer3_b   # shape:[batch_size, 1]

            return v

    def _visual_semantic_embed_reward(self, features, captions):
        reward_list = [100] * self.batch_size
        reward_list = np.asarray(reward_list, dtype="float32")
        return reward_list

    def _caption_encoder(self, time_t_captions, time_t_captions_length, num_layers):
        """
            This function is to implement encode caption and get st.and the output is the last state of RNN Cell
        Args:
            time_t_captions: random selected caption of original caption and it do not need  embed,
                            so it's shape  is (batch_size, max_sentence_length)
            time_t_captions_length: real length of each random selected captions
            num_layers: if num_layers equal to 1,we just use single rnn here, else we would MultiRNN
        Returns:
            st: hidden state of RNN_Cell, of which shape is [batch_size, hidden_units]
        """
        encoder_embed_inputs = tc.layers.embed_sequence(time_t_captions, self.vocab_size, self.embed_size)

        rnn_cell = self.cell(self.hidden_units)

        if num_layers == 1:
            _, encoder_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                 inputs=encoder_embed_inputs,
                                                 sequence_length=time_t_captions_length,
                                                 dtype=tf.float32)
            return encoder_state
        else:
            cell = tc.rnn.MultiRNNCell([rnn_cell for _ in range(num_layers)])
            _, encoder_state = tf.nn.dynamic_rnn(cell=cell,
                                                 inputs=encoder_embed_inputs,
                                                 sequence_length=time_t_captions_length,
                                                 dtype=tf.float32)
            return encoder_state

    def build_model(self):
        features = self.features
        captions = self.captions
        captions_length = self.captions_length
        original_caption = self.original_caption

        # batch_norm
        features = self._batch_norm(features, mode='train', name='conv_features')

        # shape:[batch_size, 512]
        features_proj = self._proj_features(features)

        # shape:[batch_size, 512]
        # encode caption use GRU
        st = self._caption_encoder(captions, captions_length, self.num_layers)

        # shape:[batch_size, 1024]
        concat_features = tf.concat([features_proj, st], axis=1)

        # value prediction of value network
        v = self._mlp_layer(concat_features)

        # get visual-semantic embedding reward
        reward = self._visual_semantic_embed_reward(features, original_caption)

        loss = self._loss(v, reward)

        return loss


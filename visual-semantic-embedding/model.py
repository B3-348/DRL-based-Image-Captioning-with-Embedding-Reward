import tensorflow as tf
import numpy as np

class VS_embedding(object):
    def __init__(self, batch_size, feature_dim, hidden_dim, vocab_num, embedding_dim,
                 hidden_unit, max_len, margin):
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.vocab_num = vocab_num
        self.embedding_dim = embedding_dim
        self.hidden_unit = hidden_unit
        self.max_len = max_len
        self.margin = margin
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.feature_dim])
        self.captions = tf.placeholder(tf.int32, [self.batch_size, self.max_len])
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.W1 = tf.get_variable('W1', [self.feature_dim, self.hidden_dim], initializer=self.weight_initializer)
        self.b1 = tf.get_variable('b1', [self.hidden_dim], initializer=self.const_initializer)
        self.W2 = tf.get_variable('W2', [self.hidden_unit * 2, self.hidden_dim], initializer=self.weight_initializer)
        self.b2 = tf.get_variable('b2', [self.hidden_dim], initializer=self.const_initializer)
        self.embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab_num, self.embedding_dim],
                                           initializer=self.emb_initializer)
    def _word_embedding(self, inputs, reuse = False):
        #with tf.variable_scope('word_embedding', reuse=reuse):
            #embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab_num, self.embedding_dim], initializer=self.emb_initializer)
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs, name='word_vector')  # (N, T, M) or (N, M)
        return embedded

    def _image_encoder(self):
        # with tf.variable_scope('image_encoder'):
        #     W1 = tf.get_variable('W1', [self.feature_dim, self.hidden_dim], initializer=self.weight_initializer)
        #     b1 = tf.get_variable('b1', [self.hidden_dim], initializer=self.const_initializer)
        img_encode = tf.nn.tanh(tf.matmul(self.images, self.W1) + self.b1)
        return img_encode

    def _caption_encoder(self):
        with tf.variable_scope('caption_encoder_encoder', reuse=tf.AUTO_REUSE):
            inputs = self._word_embedding(self.captions)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.hidden_unit, state_is_tuple = False)
            h0 = lstm_cell.zero_state(self.batch_size, np.float32)
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=h0)
            #tf.reshape()
            #W2 = tf.get_variable('W2', [self.hidden_unit*2, self.hidden_dim], initializer=self.weight_initializer)
            #b2 = tf.get_variable('b2', [self.hidden_dim], initializer=self.const_initializer)
            cap_encode = tf.nn.tanh(tf.matmul(state, self.W2) + self.b2)
            return cap_encode

    def _loss(self):
        img_encode = self._image_encoder()
        cap_encode = self._caption_encoder()
        scores_matrix = tf.matmul(img_encode,tf.transpose(cap_encode))
        diagonal = tf.diag_part(scores_matrix)
        cost_cap = tf.maximum(0.0, self.margin - diagonal + scores_matrix)
        diagonal = tf.reshape(diagonal, [-1, 1])
        cost_img = tf.maximum(0.0, self.margin - diagonal + scores_matrix)
        cost_cap = tf.matrix_set_diag(cost_cap, [0]*self.batch_size)
        cost_img = tf.matrix_set_diag(cost_img, [0]*self.batch_size)
        loss = tf.reduce_sum(cost_img) + tf.reduce_sum(cost_cap)
        return loss

import tensorflow as tf
import tensorflow.contrib as tc
import data_utils
import  numpy as np
import policy_config as config
import os
class RNN_p():
    def __init__(self,word_to_idx):
        self.end = 2
        self.pad = 0
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.vocab_path = config.vocab_path
        self.vocab_size = len(self.word_to_idx)
        self.embed_size = config.embed_size
        self.n_time_step = config.n_time_step
        self.img_features_size = config.img_feature_size#4096
        self.input_size = config.input_size#512
        self.hidden_size = config.hidden_size #512
        self.batch_size = config.batch_size

        self.weight_initializer = tc.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.1)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0,maxval=1.0)
        self.embed_matric = self.init_embed()

        #place holder for features and captions
        self.features = tf.placeholder(tf.float32, [self.batch_size,self.img_features_size])
        self.captions = tf.placeholder(tf.int32, [self.batch_size,  self.n_time_step+1])

        self.test_features = tf.placeholder(tf.float32,[self.batch_size,self.img_features_size])



    def init_lstm(self,features,reuse = False):
        with tf.variable_scope('init_lstm',reuse=reuse):
            #init h
            l_w_h = tf.get_variable('l_w_h',[self.img_features_size,self.hidden_size],initializer=self.weight_initializer)
            l_b_h = tf.get_variable('l_b_h',[self.hidden_size],initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features,l_w_h)+l_b_h)

            #init c
            l_w_c = tf.get_variable('l_w_c',[self.img_features_size,self.hidden_size],initializer=self.weight_initializer)
            l_b_c = tf.get_variable('l_b_c',[self.hidden_size],initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features,l_w_c)+l_b_c)

            #init x
            l_w_x = tf.get_variable('l_w_x',[self.img_features_size,self.input_size],initializer=self.weight_initializer)
            l_b_x = tf.get_variable('l_b_x',[self.input_size],initializer=self.const_initializer)
            init_input_x = tf.nn.relu(tf.matmul(features,l_w_x) +l_b_x)

            return init_input_x,h,c


    def init_rnn(self,features,reuse = False):#feature_shape:batch_size * 4096
        """

        :param features:
        :return:
        """
        with tf.variable_scope('init_rnn',reuse = reuse):
            #####initial input x0
            w_image = tf.get_variable('w_image',[self.img_features_size,self.input_size],initializer=self.weight_initializer)
            b_image = tf.get_variable('b_image',[self.input_size],initializer=self.const_initializer)
            init_input_x = tf.nn.relu(tf.matmul(features,w_image) +b_image)

            #initial hidden state h0
            #features_sum =tf.expand_dims( tf.reduce_sum(features,0),0)
            init_h_w = tf.get_variable('init_h_w',[self.img_features_size,self.hidden_size],initializer=self.weight_initializer)
            init_h_b = tf.get_variable('init_h_b',[self.hidden_size],initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features,init_h_w) + init_h_b)

            return init_input_x, h

    def init_embed(self,reuse = False):
        with tf.variable_scope('init_embed',reuse = reuse):
            embed = tf.get_variable('embed',[self.vocab_size,self.embed_size],initializer=self.emb_initializer)
            return embed


    def word_embedding(self,inputs,embed):
        """

        :param inputs:
        :param reuse:
        :return:
        """

        #shape:#[batch_size ,caption_len , embed_size]
        embedding_input = tf.nn.embedding_lookup(embed, inputs, name='word_vector')

        return embedding_input

    def probility_policy(self,h,reuse = False):#h shape:[batch_size ,hidden_size]
        """

        :param h:
        :return:
        """
        with tf.variable_scope('pro_policy',reuse = reuse):
            # shape:[batch_size, embed_size]
            pro_w = tf.get_variable('pro_w',[self.hidden_size,self.embed_size],initializer=self.weight_initializer)
            pro_b = tf.get_variable('pro_b',[self.embed_size],initializer=self.const_initializer)
            logits = tf.matmul(h,pro_w)+pro_b
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits, 0.5)

            # shape [batch_size , vocab]
            pro_word_w = tf.get_variable('pro_word_w',[self.embed_size,self.vocab_size],initializer=self.weight_initializer)
            pro_word_b = tf.get_variable('pro_word_b',[self.vocab_size],initializer=self.const_initializer)
            logits_word = tf.matmul(logits,pro_word_w)+pro_word_b

            return  logits_word

    def loss(self,logit, captions, t):
        """

        :param logit:
        :param captions:
        :param t:
        :return:
        """
        captions_out = captions[:,:]
        #shape:[batch_size,n_time_step]
        mask = tf.to_float(tf.not_equal(captions_out,self.end or self.pad))
        #shape:[batch_size,1]
        labels = tf.expand_dims(captions[: ,t],1)
        #shape:[batch_size,1]
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
        #shape:[batch_size.2]
        concated = tf.concat([indices,labels], 1)
        #shape:[batch,vocab]
        onehot_labels = tf.sparse_to_dense(concated , tf.stack([self.batch_size,self.vocab_size]),1.0,0.0)
        #@print(onehot_labels.shape)
        #print(logit.shape)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=onehot_labels)*mask[:,t])
        loss = loss/tf.to_float(self.batch_size)

        return loss
    def batch_norm(self, x, mode='train',name=None):
        """

        :param x:
        :param model:
        :param name:
        :return:
        """
        return tc.layers.batch_norm(inputs=x,
                                    decay=0.95,
                                    center=True,
                                    scale=True,
                                    is_training=(mode=='train'),
                                    updates_collections=None,
                                    scope=(name+'batch_norm')
                                   )
    def get_input_x(self, logit,reuse = False):
        """

        :param logit:shape:[batch_size,1]
        :return:
        """
        with tf.variable_scope('get_input_x',reuse= reuse):
            #shape:[batch,1]
            #word_idx = tf.arg_max(logit, dimension=1)
            #shape:[batch_size,embed_size]
            embed_word = self.word_embedding(logit,self.embed_matric)#shape:[batch_size,embed_size]

            input_w = tf.get_variable('input_w',[self.embed_size,self.input_size],initializer=self.weight_initializer)
            input_b = tf.get_variable('input_b',[self.input_size],initializer=self.const_initializer)
            input_x = tf.nn.relu(tf.matmul(embed_word,input_w)+input_b)
            #shape:[batch_size,input_size]

            return  input_x
    def test_input_x(self, logit,reuse = False):
        """

        :param logit:shape:[batch_size,1]
        :return:
        """
        with tf.variable_scope('get_input_x',reuse= reuse):
            #shape:[batch,1]
            word_idx = tf.arg_max(logit, dimension=1)
            #shape:[batch_size,embed_size]
            embed_word = self.word_embedding(word_idx,self.embed_matric)#shape:[batch_size,embed_size]

            input_w = tf.get_variable('input_w',[self.embed_size,self.input_size],initializer=self.weight_initializer)
            input_b = tf.get_variable('input_b',[self.input_size],initializer=self.const_initializer)
            input_x = tf.nn.relu(tf.matmul(embed_word,input_w)+input_b)
            #shape:[batch_size,input_size]

            return  input_x

    def rnn_model(self, dropout=False):
        """

        :param input_x:
        :param h:
        :param dropout:
        :param reuse:
        :return:
        """
        features = self.features#shape:[batch_size,512]
        captions = self.captions#shape:[batch_size,512]
        captions = captions[:,1:]#shape:[batch_size,511]
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)

        #features = self.batch_norm(features,mode='train',name='conv_features')
        #word_embed = self.word_embedding(captions)
        loss = 0.0
        for ind in range(self.n_time_step):
            if ind == 0:
                x, h = self.init_rnn(features)#x.shape:[batch_size,512]   h.shape:[batch_size,512]
                pro = self.probility_policy(h,reuse=(ind!=0))
                loss = self.loss(pro,captions,ind)
                input_x = self.get_input_x(captions[ind],reuse=(ind!=0))
            else:
                with tf.variable_scope('rnn'):
                     h,o = rnn_cell(input_x,h)
                #print("h shape:%d     o shape: %d" %h.shape,o.shape)
                pro = self.probility_policy(h,reuse=(ind!=0))
                #print("pro hsape: %d" %pro.shape)
                loss += self.loss(pro, captions,ind)
                input_x = self.get_input_x(captions[ind],reuse=(ind!=0))
               # print("input s shape: %d"%input_x.shape)

        return  loss


    def test_rnn_model(self,max_len = 20):
        features = self.test_features
        #features = self.batch_norm(features,mode='test',name='conv_features')
        #input_x,h = self.init_rnn(features=features)

        pro_list = []
        index_list = []
        word_list = []
        test_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)

        for i in range(max_len):
            if i == 0:
                x,h = self.init_rnn(features)
                pro = self.probility_policy(h,reuse=(i!=0))
                index_list.append(tf.argmax(pro,1))
                input_x = self.get_input_x(pro,reuse=(i!=0))
            else:
                with tf.variable_scope('lstm'):
                    h,o = test_rnn_cell(input_x,h)
                pro = self.probility_policy(h,reuse=(i!=0))
                index_list.append(tf.argmax(pro,1))
                input_x = self.get_input_x(pro,reuse=(i!=0))

        t_idx_list = tf.transpose(tf.stack(index_list),(1,0))

        return t_idx_list

    def lstm_model(self, dropout=False):
        """

        :param input_x:
        :param h:
        :param dropout:
        :param reuse:
        :return:
        """
        features = self.features  # shape:[batch_size,512]
        captions = self.captions  # shape:[batch_size,512]
        captions = captions[:, 1:]  # shape:[batch_size,511]
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

        features = self.batch_norm(features,mode='train',name='conv_features')
        # word_embed = self.word_embedding(captions)
        loss = 0.0
        for ind in range(self.n_time_step):
            if ind == 0:
                x, h,c = self.init_lstm(features)  # x.shape:[batch_size,512]   h.shape:[batch_size,512]
                pro = self.probility_policy(h, reuse=(ind != 0))
                loss = self.loss(pro, captions, ind)
                input_x = self.get_input_x(captions[:,ind], reuse=(ind != 0))
            else:
                with tf.variable_scope('lstm'):
                    _,(c,h) = lstm_cell(input_x, [c,h])
                # print("h shape:%d     o shape: %d" %h.shape,o.shape)
                pro = self.probility_policy(h, reuse=(ind != 0))
                # print("pro hsape: %d" %pro.shape)
                loss += self.loss(pro, captions, ind)
                input_x = self.get_input_x(captions[:,ind], reuse=(ind != 0))
            # print("input s shape: %d"%input_x.shape)

        return loss

    def test_lstm_model(self,max_len = 20):
        print("go to lstm !")
        features = self.test_features
        features = self.batch_norm(features,mode='test',name='conv_features')
        test_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        index_list = []
        print("innier feature num : " ,features.shape[0])
        for i in range(max_len):
            if i == 0:
                x,h,c = self.init_lstm(features)
                pro = self.probility_policy(h,reuse=(i!=0))
                index_list.append(tf.argmax(pro,1))
                input_x = self.test_input_x(pro,reuse=(i!=0))
            else:
                with tf.variable_scope('lstm'):
                    _,(c,h) = test_lstm_cell(input_x,state=[c,h])
                pro = self.probility_policy(h,reuse=(i!=0))
                word_idx = tf.argmax(pro,1)
                index_list.append(word_idx)
                input_x = self.test_input_x(pro,reuse=(i!=0))

        t_idx = tf.transpose(index_list,(1,0))
        print("inner idx num: ",t_idx.get_shape().as_list())

        return t_idx




















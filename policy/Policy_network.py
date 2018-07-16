import tensorflow as tf
import os
import  numpy as np
from RNN_p import RNN_p
import VGG16
from scipy import ndimage
import  pickle
import policy_config as config
import data_utils
import time
import matplotlib.pyplot as plt
import skimage.transform

class policy_network(object):
    def __init__(self,rnn_model,data):
        self.rnn_model = rnn_model
        self.data = data
        self.vgg_parap_path=config.vgg_para_path
        self.img_path = config.img_path
        self.feature_path = config.features_path
        self.cap_path = config.caption_path
        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.model_path = config.model_path
        self.optimizer = tf.train.AdadeltaOptimizer
        self.log_path = config.log_path


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def train(self,pretrained_model_path=None):
        """

        :return:
        """
        features = self.data['features']
        seqcaptions = self.data['captions']
        img_idx = self.data['image_idxs']
        n_iter_per_epoch =int(features.shape[0]/self.batch_size)
        num_img = features.shape[0]

        with tf.variable_scope(tf.get_variable_scope()):
            t_loss = self.rnn_model.lstm_model()
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            train_op = self.optimizer(learning_rate=self.learning_rate).minimize(t_loss)
            '''
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss,tf.trainable_variables())
            grads_and_vars = list(zip(grads,tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            '''

        '''
        #summary op
        tf.summary.scalar('batch loss',loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name,var)
        for grad,var in grads_and_vars:
            tf.summary.histogram(var.op.name+'/gradient',grad)
        summary_op = tf.summary.merge_all()
        '''


        epoch_loss = 0.0
        start_t = time.time()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.summary.FileWriter(self.log_path,graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=20)

            if pretrained_model_path is not None:
                print("start traing with pretrained model..")
                saver.restore(sess,pretrained_model_path)
                print("retore end!")

            for epoch in range(self.n_epochs):
                rand_idx = np.random.permutation(num_img)
                seqcaptions = seqcaptions[rand_idx]
                img_idx = img_idx[rand_idx]

                for i in range(n_iter_per_epoch):
                    captions_batch = seqcaptions[i*self.batch_size:(i+1)*self.batch_size]
                    img_idx_batch = img_idx[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[img_idx_batch]

                    feed_dict = {rnn_model.features:features_batch,rnn_model.captions:captions_batch}
                    _,loss = sess.run([train_op,t_loss],feed_dict)
                    epoch_loss += loss
                    '''
                    if i%10==0:
                        summary = sess.run(summary_op,feed_dict)
                        summary_writer.add_summary(summary,epoch*n_iter_per_epoch+i)
                    '''

                    print("Epoch : %d , batch: %d   " % (epoch, i))
                    print("Curent Loss :", loss)
                    print("elpapsed time :  ", time.time()-start_t)

                print("Epoch : %d  , Epoch loss : " ,  epoch,epoch_loss)
                epoch_loss = 0.0
                saver.save(sess,self.model_path,global_step=epoch+1)
                print("   Epoch : %d  model saved!" %epoch)

    def test(self,data,test_path,split='test',visual=True,save_test_caption=True):
        """
        :param data:
        :param split:
        :return:
        """
        #features = data['features']
        test_idx_caption  = self.rnn_model.test_lstm_model()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            print("start test with pretrained model..")
            saver.restore(sess, test_path)
            print("retore end!")
            features_batch,img_files = data_utils.sample_mini_features(data,self.batch_size)
            print("features shape:",features_batch.shape[0])
            feed_dict = {self.rnn_model.test_features:features_batch}
            idx_seq = sess.run([test_idx_caption],feed_dict)
            print(idx_seq[0])
            decode_caption = data_utils.docode_idx_to_caption(idx_seq,self.rnn_model.idx_to_word)
            for i in range(len(decode_caption)):
                print(len(decode_caption[i].split()))
                print(decode_caption[i])

            if visual:
                for n in range(10):
                    print("test caption: %s" %decode_caption[n])

                    #plot original image
                    img = ndimage.imread('/home/cy/'+img_files[n])
                    plt.subplot(4,5,1)
                    plt.text(0,1,decode_caption[n])
                    plt.imshow(img)
                    plt.axis('off')



    def get_img_features(self,img_path_list,feature_path):
        """

        :param img_path:
        :param feature_path:
        :return:
        """
        num_img = len(img_path_list)
        vgg = VGG16.Vgg16(self.vgg_parap_path)
        vgg.build()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            all_features = np.array([num_img,4096],dtype=np.float)
            for start,end in zip(range(0,num_img,self.batch_size), range(self.batch_size,num_img,self.batch_size)):
                img_batch_file = img_path_list[start:end]
                img_batch = np.array(list(map(lambda x:ndimage.imread(x,mode='RGB'), img_batch_file))).astype(np.float32)
                features = sess.run(vgg.prob, feed_dict={vgg.images:img_batch})
                print(features)
                all_features[start:end,:] = features

            with open(feature_path,'wb') as f:
                pickle.dump(all_features,f,pickle.HIGHEST_PROTOCOL)
                print("Save %s..." %feature_path)
        return all_features



data = data_utils.load_coco_data(config.data_path,split='test')
rnn_model = RNN_p(word_to_idx=data['word_to_idx'])
policy=policy_network(rnn_model,data)
#policy.train(pretrained_model_path=config.model_path+"-9")
policy.test(data,test_path=config.model_path+"-621")




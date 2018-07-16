import tensorflow as tf
import numpy as np
from utils import load_coco_data
import os
import homogeneous_data
from vocab import build_dictionary

from datasets import load_dataset
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
SAVE_EVERY = 10
model_paths = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/VS-embedding/model/model-130'

class slover(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def train(self):
        #data = 'f8k'
        #print('Loading dataset')
        #train, dev = load_dataset(data)[:2]


        # Create and save dictionary
        #print('Creating dictionary')
        #worddict = build_dictionary(train[0] + dev[0])[0]
        #n_words = len(worddict)
        #model_options['n_words'] = n_words
        #print('Dictionary size: ' + str(n_words))

        # with open('%s.dictionary.pkl' % saveto, 'wb') as f:
        #     pkl.dump(worddict, f)

        # Inverse dictionary
        # word_idict = dict()
        # for kk, vv in worddict.items():
        #     word_idict[vv] = kk
        # word_idict[0] = '<eos>'
        # word_idict[1] = 'UNK'
        model_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/VS-embedding/model'
        n_epochs = 100000
        #train_iter_data = homogeneous_data.HomogeneousData([train[0], train[1]], batch_size=BATCH_SIZE, maxlen=100)
        #n_examples = len(train[0])
        n_examples = self.data['features'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / BATCH_SIZE))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        with tf.variable_scope(tf.get_variable_scope()):
            loss, similarity, wrong_smi = self.model._loss()
            #similarity = self.model._similarity()

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=40)
            for e in range(n_epochs):
                losses = 0.0
                #i=0
                #for captions, features in train_iter_data:
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]
                    #x, mask, im = homogeneous_data.prepare_data(captions, features, worddict, maxlen=100, n_words=n_words)

                for i in range(n_iters_per_epoch - 1):
                    captions_batch = captions[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    #features_batch = features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    image_idxs_batch = image_idxs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    features_batch = features[image_idxs_batch]

                    feed_dict = {self.model.images: features_batch, self.model.captions: captions_batch}
                    _, batch_loss, batch_similarity, batch_wrong_smi= sess.run([train_op, loss, similarity, wrong_smi], feed_dict)
                    losses += batch_loss
                    if(i==n_iters_per_epoch-2):
                        print('epoch %d simi:' % e)
                        print(batch_similarity)
                        print(batch_wrong_smi)
                print('epoch %d loss %s:' % (e, losses / (n_iters_per_epoch)))


                if (e + 1) % SAVE_EVERY == 0:
                    saver.save(sess, os.path.join(model_path, 'model'), global_step=e + 1)
                    print("model-%s saved." % (e + 1))

    def _similarity(self, img_encode, cap_encode):
        inner_product = tf.reduce_sum(tf.multiply(img_encode, cap_encode), axis=1)
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(img_encode), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(cap_encode), axis=1))
        cos = inner_product / (norm1 * norm2)
        return cos

    def calculate_reward(self, img_features, cap_tokens, model_path = model_paths):
        with tf.variable_scope(tf.get_variable_scope()):
            loss, similarity, wrong_smi = self.model._loss()
            #img_encoder = self.model._image_encoder()
            #cap_encoder = self.model._caption_encoder()
            # tf.get_variable_scope().reuse_variables()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            feed_dict = {self.model.images: img_features, self.model.captions: cap_tokens}
            #img_encoder, cap_encoder = sess.run([img_encoder, cap_encoder], feed_dict)
            l, si, ws = sess.run([loss, similarity, wrong_smi], feed_dict)
            #reward = sess.run(self._similarity(img_encoder, cap_encoder))
            return si


import tensorflow as tf
import random
import numpy as np
import time
import os
from vse.model import VS_embedding
from vse.solver import slover


class ValueNetWordSolver(object):
    def __init__(self, word2idx, model, data, val_data, batch_size=100, n_epochs=20000,
                 save_every=10, print_every=100, model_path=None, pretrained_model=None):
        self.word2idx = word2idx
        self.model = model
        self.data = data
        self.val_data = val_data
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.save_every = save_every
        self.print_every = print_every

        self.model_path = model_path
        self.pretrained_model = "/home/lemin/1TBdisk/PycharmProjects/DRL-based-Image-Captioning-with-Embedding-Reward/vnw/vnw_model/vs_model-180"

        self._start = word2idx['<START>']
        self._null = word2idx['<NULL>']
        self._eos = word2idx['<END>']

    def _sample_time_step(self, caption):
        """
            This function is to sample a time t step of caption,
            for example if caption is [[1,4,5,6,2,0],
                                       [1,12,22,5,3,2]
                                       ]
            The real caption is [3,4] so maybe the sample can be [2,3]
            Thus, we can get [[4,5],[12,22,5]] which is encoder's input of RNN
        Args:
            caption: a batch_size of caption, shape:[batch_size, max_time_step]
        Returns:
            sample_caption:for example:[[4,5],[12,22,5]]
            sample_caption_len:[2,3]
            caption: original caption
        """
        # change to python list
        caption = np.ndarray.tolist(caption)

        # get real len
        caption_len = list(map(lambda x: x.index(self._eos) - 1, caption))

        # random select
        random_select = list(map(lambda x: random.randint(1, x), caption_len))
        sample_caption_len = np.array(random_select).astype("int32")

        sample_captions = []
        for index, sentence in enumerate(caption):
            sample_sentence = sentence[1:sample_caption_len[index] + 1]
            sample_captions.append(sample_sentence)

        # change back to numpy list in order to can be feat into tensorflow
        sample_captions = np.asarray(self._pad_sentence_batch(sample_captions), dtype="int32")

        return sample_captions, sample_caption_len, caption

    def _pad_sentence_batch(self, sentence_batch):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [self._null] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def train(self):
        n_examples = self.data['features'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_op = self.model.train_op(loss)

        print("The number of epoch: %d" % self.n_epochs)
        print("Data size: %d" % n_examples)
        print("Batch size: %d" % self.batch_size)
        print("Iterations per epoch: %d" % n_iters_per_epoch)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=10)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss_list = []
            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idx = np.random.permutation(n_examples)
                captions = captions[rand_idx]
                image_idxs = image_idxs[rand_idx]
                calculate_time = 0

                s_captions, s_captions_length, o_captions = self._sample_time_step(captions)

                for i in range(n_iters_per_epoch):
                    o_captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    captions_batch = s_captions[i*self.batch_size:(i+1)*self.batch_size]

                    captions_length_batch = s_captions_length[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]

                    start_time = time.time()
                    vs_graph = tf.Graph()
                    # calculate reward

                    model = VS_embedding(
                        batch_size=100,
                        feature_dim=4096,
                        hidden_dim=1024,
                        vocab_num=10000,
                        embedding_dim=512,
                        hidden_unit=1024,
                        max_len=17,
                        margin=0.2,
                        graph=vs_graph
                    )

                    solver = slover(model, vs_graph)

                    batch_reward = solver.calculate_reward(features_batch, o_captions_batch)
                    end_time = time.time()
                    calculate_time += end_time - start_time

                    feed_dict = {self.model.features: features_batch,
                                 self.model.captions: captions_batch,
                                 self.model.captions_length: captions_length_batch,
                                 self.model.reward: batch_reward}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    if (i+1) % 100 == 0:
                        print("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e+1, i+1, l))

                print("Previous epoch loss: ", prev_loss)
                print("Current epoch loss: ", curr_loss)
                print("Elapsed time: ", time.time() - start_t)

                # if loss is no decrease, we will use learning_rate decay
                if len(prev_loss_list) > 2 and curr_loss > max(prev_loss_list[-3:]):
                    sess.run(self.model.learning_rate_decay_op)

                prev_loss = curr_loss
                prev_loss_list.append(prev_loss)
                curr_loss = 0

                # save vnw_model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'vs_model'), global_step=e+1)
                    print("vs_model-%s saved." % (e+1))
                print(calculate_time)

    def test(self, data):
        features = data["features"]
        captions = data["captions"]

        captions_length = np.asarray([len(caption) for caption in captions]).astype("int32")
        captions = self._pad_sentence_batch(captions)

        with tf.variable_scope(tf.get_variable_scope()):
            v = self.model.build_sampler()
            tf.get_variable_scope().reuse_variables()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)
            else:
                print("There is no model!")
                raise Exception("If you want to test your model, you need to have a model first!")
            feed_dict = {
                self.model.features: features,
                self.model.captions: captions,
                self.model.captions_length: captions_length
            }
            v = sess.run(v, feed_dict=feed_dict)

            print(v)
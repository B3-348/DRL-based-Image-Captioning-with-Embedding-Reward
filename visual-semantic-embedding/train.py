import tensorflow as tf
import numpy as np
import model
from utils import load_coco_data
import os
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SAVE_EVERY = 10
model_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/DRL-based-Image-Captioning-with-Embedding-Reward/model/model-200'

def train(data):
    model_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/DRL-based-Image-Captioning-with-Embedding-Reward/model'
    n_epochs = 100000
    n_examples = data['features'].shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples)/BATCH_SIZE))
    features = data['features']
    captions = data['captions']
    image_idxs = data['image_idxs']
    #val_features = val_data['features']
    #n_iters_val = int(np.ceil(float(val_features.shape[0]) / BATCH_SIZE))
    VS_embedding = model.VS_embedding(
        batch_size = 64,
        feature_dim = 4096,
        hidden_dim = 1024,
        vocab_num = 25000,
        embedding_dim = 300,
        hidden_unit = 512,
        max_len = 17,
        margin = 0.2
    )
    with tf.variable_scope(tf.get_variable_scope()):
        loss = VS_embedding._loss()
        # img = VS_embedding._image_encoder()
        # cap = VS_embedding._caption_encoder()

    with tf.variable_scope(tf.get_variable_scope(), reuse = False):
        train_op = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep = 40)
        for e in range(n_epochs):
            rand_idxs = np.random.permutation(n_examples)
            captions = captions[rand_idxs]
            image_idxs = image_idxs[rand_idxs]
            losses = 0.0
            for i in range(n_iters_per_epoch-1):
                captions_batch = captions[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                image_idxs_batch = image_idxs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                features_batch = features[image_idxs_batch]
                feed_dict = {VS_embedding.images: features_batch, VS_embedding.captions: captions_batch}
                _, batch_loss = sess.run([train_op, loss], feed_dict)
                losses += batch_loss
            print('epoch %d loss %s:' %(e, losses))

            if (e + 1) % SAVE_EVERY == 0:
                saver.save(sess, os.path.join(model_path, 'model'), global_step=e + 1)
                print("model-%s saved." % (e + 1))

def cos(img_encode, cap_encode):
    inner_product = tf.reduce_sum(tf.multiply(img_encode, cap_encode), axis=1)
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(img_encode), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(cap_encode), axis=1))
    cos = inner_product / (norm1 * norm2)
    return cos

def calculate_reward(VS_embedding, img_features, cap_tokens, model_path):
    # VS_embedding = model.VS_embedding(
    #     batch_size=64,
    #     feature_dim=4096,
    #     hidden_dim=1024,
    #     vocab_num=25000,
    #     embedding_dim=300,
    #     hidden_unit=512,
    #     max_len=17,
    #     margin=0.2
    # )
    with tf.variable_scope(tf.get_variable_scope()):
        img_encoder = VS_embedding._image_encoder()
        cap_encoder = VS_embedding._caption_encoder()
        #tf.get_variable_scope().reuse_variables()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        feed_dict = {VS_embedding.images: img_features, VS_embedding.captions: cap_tokens}
        img_encoder, cap_encoder = sess.run([img_encoder, cap_encoder], feed_dict)
        reward = sess.run(cos(img_encoder, cap_encoder))
        return reward
if __name__ == '__main__':
    data_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/show-attend-and-tell-master/data'
    data = load_coco_data(data_path)
    train(data)
    # VS_embedding = model.VS_embedding(
    #     batch_size=64,
    #     feature_dim=4096,
    #     hidden_dim=1024,
    #     vocab_num=25000,
    #     embedding_dim=300,
    #     hidden_unit=512,
    #     max_len=17,
    #     margin=0.2
    # )
    # features = data['features']
    # captions = data['captions']
    # img = features[:64]
    # cap = captions[:64]
    # cap1 = captions[128:192]
    # correct = calculate_reward(VS_embedding,img, cap, model_path)
    # wrong = calculate_reward(VS_embedding,img,cap1, model_path)
    # print(correct,'\n', wrong)

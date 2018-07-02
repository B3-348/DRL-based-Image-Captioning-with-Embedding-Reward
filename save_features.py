import tensorflow as tf
import numpy as np
import pickle
from scipy import ndimage
from vgg16 import Vgg16

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    vgg_model_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/data/vgg16.npy'
    vggnet = Vgg16(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        save_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/show-attend-and-tell-master/data/train/4096feats.pkl'
        anno_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/show-attend-and-tell-master/data/train/train.annotations.pkl'
        annotations = load_pickle(anno_path)
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)
        batch_size = 64
        pre_image_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/show-attend-and-tell-master/'
        image_path = [pre_image_path + image for image in image_path[:n_examples]]
        #image_batch_file = image_path[:n_examples]
        #image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(
        #   np.float32)
        #feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
        #print(feats)
        all_feats = np.ndarray([n_examples, 4096], dtype=np.float32)
        for start, end in zip(range(0, n_examples, batch_size),
                              range(batch_size, n_examples + batch_size, batch_size)):
            image_batch_file = image_path[start:end]
            image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(
                np.float32)
            feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
            all_feats[start:end, :] = feats
            print("Processed %d %s features.." % (end, 'test'))
        save_pickle(all_feats, save_path)
        print ("Saved %s.." % (save_path))
        # all_feats[start:end, :] = feats
        # print("Processed %d %s features.." % (end, split))
        # for split in ['train', 'val', 'test']:
        #     anno_path = './data/%s/%s.annotations.pkl' % (split, split)
        #     save_path = './data/%s/%s.features.pkl' % (split, split)
        #     annotations = load_pickle(anno_path)
        #     image_path = list(annotations['file_name'].unique())
        #     n_examples = len(image_path)
        #     print(n_examples)
        #
        #     all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
        #
        #     for start, end in zip(range(0, n_examples, batch_size),
        #                           range(batch_size, n_examples + batch_size, batch_size)):
        #         image_batch_file = image_path[start:end]
        #         image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(
        #             np.float32)
        #         feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
        #         all_feats[start:end, :] = feats
        #         print ("Processed %d %s features.." % (end, split))
        #
        #     # use hickle to save huge feature vectors
        #     save_pickle(all_feats, save_path)
        #     print ("Saved %s.." % (save_path))
if __name__ == '__main__':
    main()
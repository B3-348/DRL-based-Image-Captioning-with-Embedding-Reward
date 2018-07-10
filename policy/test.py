import re
import pickle
import os
import time
import numpy as np

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' %path)
        return file


def load_coco_data(data_path='./f30kdata', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
    with open(os.path.join(data_path, '%s.features.pkl' % split), 'rb') as f:
        data['features'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.file.names.pkl' % split), 'rb') as f:
        data['file_names'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.captions.pkl' % split), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.pkl' % split), 'rb') as f:
        data['image_idxs'] = pickle.load(f)

    if split == 'train':
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)

    for k, v in data.items():
        if type(v) == np.ndarray:
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), len(v))
    end_t = time.time()
    print("Elapse time: %.2f" % (end_t - start_t))
    return data
data_path = os.path.dirname(os.path.dirname(__file__))+'/coco_data/'
data = load_coco_data(data_path,split='train')
print("hhhh")

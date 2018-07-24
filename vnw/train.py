from vnw.model import ValueNetWork
from vnw.solver import ValueNetWordSolver
from vse.model import VS_embedding
from vse.solver import slover
import utils.data_utils as data_utils
import os
import numpy as np
import random
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    data = data_utils.load_data(data_path="../data", split="train")
    word2idx = data["word_to_idx"]

    val_data = data_utils.load_data(data_path='../data', split='val')

    model = ValueNetWork()
    solver = ValueNetWordSolver(word2idx=word2idx, model=model, data=data, val_data=val_data, batch_size=100,
                                model_path="./vnw_model")

    features = data["features"]
    captions = data["captions"]
    image_idxs = data['image_idxs']
    features = features[image_idxs]
    n_captions = []
    captions = captions[:10]
    captions = captions.tolist()

    for caption in captions:
        index = random.randint(2, 14)
        n_captions.append(caption[1:index])
    features = features[:10]
    print(n_captions[0])
    n_captions[0] = [3, 4, 5, 6, 5, 2]
    test_data = dict()
    test_data["captions"] = n_captions
    test_data["features"] = features
    solver.test(test_data)

    # vs_graph = tf.Graph()
    # # calculate reward
    #
    # model = VS_embedding(
    #     batch_size=10,
    #     feature_dim=4096,
    #     hidden_dim=1024,
    #     vocab_num=10000,
    #     embedding_dim=512,
    #     hidden_unit=1024,
    #     max_len=17,
    #     margin=0.2,
    #     graph=vs_graph
    # )
    #
    # solver = slover(model, vs_graph)
    #
    # batch_reward = solver.calculate_reward(features, captions)
    # print(batch_reward)

if __name__ == '__main__':
    main()

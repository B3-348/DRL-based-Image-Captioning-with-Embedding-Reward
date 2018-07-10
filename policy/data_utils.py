import  tensorflow as tf
import numpy as np
import json
import re
import os
import pickle
from scipy import ndimage
import VGG16
import time
from tensorflow.python.platform import gfile
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile("\d")

img_path = '/home/cy/image/train2014/'
file_path = '/home/cy/image/annotations/captions_train2014.json'
vocab_path = './vocab'
feature_path = os.path.dirname(os.path.dirname(__file__))+'/feature.pkl'
# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_START = b"_START"
_END = b"_END"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _START, _END, _UNK]
PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3

def get_captions_words(file_path):
    captions = []
    with open(file_path) as f:
        caption_data = json.load(f)
    for caption in caption_data["annotations"]:
        captions.append(caption["caption"])
    return captions


def basic_tokenizer(sentence):
    """
        Very basic tokenizer: split the sentence into a list of tokens.

    :param sentence:
    :return:
    """
    words = []
    for space_separated_gragment in sentence.strip().split():
        words.append(str(re.sub(_WORD_SPLIT,'',space_separated_gragment)))
    return [w for w in words if w]

def create_vocalbulary(vocab_path,caption_list,max_vocab_size=24462,tokenizer=None,normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from disc_data file.

        Data file is assumed to contain one sentence per line. Each sentence is
        tokenized and digits are normalized (if normalize_digits is set).
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.

        Args:
          vocabulary_path: path where the vocabulary will be created.
          captions_list: list of captions that will be used to create vocabulary.
          max_vocabulary_size: limit on the size of the created vocabulary.
          tokenizer: a function to use to tokenize each disc_data sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        Return:
            vocab_list
        """
    if not gfile.Exists(vocab_path):
        print("create vocabulary %s"% vocab_path)
        vocab = {}
        counter = 0
        for caption in caption_list:

            counter += 1
            caption=caption.lower()
            if counter % 1000 ==0:
                print("process line %d "% counter)
            #caption = tf.compat.as_bytes(caption)
            tokens = tokenizer(caption) if tokenizer else basic_tokenizer(caption)
            for w in tokens:
                word = _DIGIT_RE.sub(b"0",w) if normalize_digits else w
                if word in vocab:
                    vocab[word] +=1
                else:
                    vocab[word] =1
        vocab_list = _START_VOCAB+sorted(vocab,key=vocab.get(),reverse=True)
        if len(vocab_list) > max_vocab_size:
            vocab_list = vocab_list[:max_vocab_size]
        with gfile.GFile(vocab_path,mode='w') as vocao_file:
            for w in vocab_list:
                vocao_file.write(w + b"\n")
        return vocab_list



def load_glove(globe_path_directory,dim=100):
        """
        Args:
            glove_path: path of glove
            dim:dimension of embedding matrix
        Return:
            glove embedding matrix
        """
        word2vec = {}
        print("==> loading glove")
        with open(globe_path_directory+'/glove.6B.%s.txt'% str(dim)) as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = list(map(float,l[1:]))
        print("==>glove is loaded")
        return word2vec


def get_word_idx(vocab_list=None,vocab_path=None):
        """

        :param vocab_list:
        :param vocab_path:
        :return:
        """
        if vocab_list is not None:
            word2idx = {word:idx for idx ,word in enumerate(vocab_list)}
            idx2word=  {idx:word for idx,word in enumerate(vocab_list)}
            return word2idx,idx2word
        if gfile.Exists(vocab_path):
            vocab = []
            with gfile.GFile(vocab_path,'rb') as f:
                vocab.extend(f.readlines())
            word2idx = {word:idx for idx,word in enumerate(vocab)}
            idx2word = {idx:word for idx ,word in enumerate(vocab)}
            return  word2idx,idx2word



def sentence_to_idx(sentence,word2idx,max_sentence_len=30):
        """Convert a string to list of integers representing token-ids.

        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

        Args:
          sentence: the sentence in bytes format to convert to token-ids.
          vocabulary: a dictionary mapping tokens to integers.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.

        Returns:
          a list of integers, the token-ids for the sentence.
        """
        #shape:[1 * max_sentence_len+2]
        sentence = basic_tokenizer(sentence)
        sentence_digit = []
        #sentence_digit.append(word2idx[_START])
        if len(sentence)>max_sentence_len:
            sentence = sentence[:max_sentence_len]

        sentence_digit = [word2idx[word] for word in sentence]
        sentence_digit.append(word2idx[_END])
        if len(sentence_digit) < (max_sentence_len+1):
            for i in range(max_sentence_len+1-len(sentence_digit)):
                sentence_digit.append(word2idx[_PAD])

        return sentence_digit

def get_imgname_captions(file_path,img_path):

    img_name = os.listdir(img_path)
    id_caption = {}

    imgname_and_captions = {}
    with open(file_path)as f:
        caption_data = json.load(f)

    for caption in caption_data["annotations"]:
        id_caption[caption["image_id"]] = caption["caption"]

    for image in caption_data["images"]:
        if image['file_name'] in img_name:
            imgname_and_captions[image['file_name']] = id_caption[image['id']]

    return imgname_and_captions

def main_imgname_and_seqcaption():
        """

        :param img_name_and_captions:
        :return:
        """
        #get captions
        captions = get_captions_words(file_path)
        print(captions[0])
        #get vocabulary
        vocab_list = create_vocalbulary(vocab_path,captions)
        # get word2idx
        word2idx,idx2word = get_word_idx(vocab_list,vocab_path)
        #get image name and captions
        img_name_and_captions  = get_imgname_captions(file_path.img_path)
        ima_path_and_seqsentence = {}
        ima_path_list = []
        seqcaptions_list = []
        for key in img_name_and_captions.keys():
            ima_path_list.append(img_path+key)
            seqcaptions_list.append(sentence_to_idx(img_name_and_captions[key],word2idx))


        return ima_path_list,seqcaptions_list


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

def sample_mini_features(data,batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size,batch_size)
    features = data['features'][mask]
    file_name = data['file_names'][mask]
    #true_caption = []
    return  features,file_name

def docode_idx_to_caption(idx_seq,idx_to_word):
    '''

    :param idx_seq:
    :param idx_to_word:
    :return:
    '''
    """
    decode = []
    print(len(idx_seq))
   # print(idx_seq.shape[0])
    len_seq = len(idx_seq)
    for seq in range(len_seq):
        sentence = []
        for idx in idx_seq[seq].shape:
            word  = idx_to_word[idx_seq[seq][idx]]
            if word == '<END>':
                sentence.append('.')
            if word != '<NULL>':
                sentence.append(word)
        decode.append(' '.join(sentence))
    return decode
    """

    idx_seq = idx_seq[0]
    if idx_seq.ndim == 1:
        t = idx_seq.shape[0]
        n = 1
    else:
        print(idx_seq.shape)
        n,t = idx_seq.shape

    decode = []
    for i in range(n):
        sentence = []
        for  j in range(t):
            if idx_seq.ndim ==1:
                word = idx_to_word[idx_seq[j]]
            else:
                word = idx_to_word[idx_seq[i][j]]
            if word == '<END>':
                sentence.append('.')
            if word != '<NULL>':
                sentence.append(word)
        decode.append(' '.join(sentence))
    return decode





#ima_path_list,seqcaption_list = main_imgname_and_seqcaption()
#get_img_features(ima_path_list,feature_path)










                    




import os

learning_rate=0.01
n_epochs=10000
features_path=os.path.dirname(os.path.dirname(__file__))+'/train.feayures.pkl'
img_path = '/home/cy/image/train2014/'
vgg_para_path = '/home/cy/vgg16.npy'
caption_path = '/home/cy/image/annotations/captions_train2014.json'
data_path = os.path.dirname(os.path.dirname(__file__))+'/coco_data/'
batch_size = 100
model_path = os.path.dirname(os.path.dirname(__file__))+'/model/'
n_time_step = 16
log_path=os.path.dirname(os.path.dirname(__file__))+'/log/'
img_path = '/home/cy/image/train2014/'
file_path = '/home/cy/image/annotations/captions_train2014.json'
vocab_path = os.path.dirname(os.path.dirname(__file__))+'/vocab'
input_size = 512
hidden_size= 512
embed_size =100
img_feature_size=4096

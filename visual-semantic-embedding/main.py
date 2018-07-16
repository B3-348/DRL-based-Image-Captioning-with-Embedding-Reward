from model import VS_embedding
from solver import slover
from utils import load_coco_data
if __name__ == '__main__':

    model = VS_embedding(
        batch_size=128,
        feature_dim=4096,
        hidden_dim=1024,
        vocab_num=25000,
        embedding_dim=300,
        hidden_unit=512,
        max_len=17,
        margin=0.2
    )
    data_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/workspace/show-attend-and-tell-master/data'
    data = load_coco_data(data_path)
    slover = slover(model, data)
   #slover.train()
    i=650
    cap = data['captions'][i:i+128]
    image_idxs = data['image_idxs'][i:i+128]
    img = data['features'][image_idxs]
    reward = slover.calculate_reward(img, cap)
    print(reward)
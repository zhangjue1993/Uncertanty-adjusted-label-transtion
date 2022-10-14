import os, json,argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, help='config path')
args = parser.parse_args()
file_dir = args.config_path
with open(file_dir) as f:
        config = json.load(f)
print(config)
img_size = config['DATA']['image_size']
bs = config['TRAIN']['batch_size']
train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
train_bayes_label = config['DATA']['bayes_optimal_label']
sigma_path = config['DATA']['train_sigma_dir']
test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])
#test_sigma = config['DATA']['test_sigma_dir']


file_dir = os.listdir(train_label)



for file_ in file_dir:

    img_path = os.path.join(train_img, file_)
    label_path = os.path.join(train_label, file_)
    bayes_label_path = os.path.join(train_bayes_label, file_)
    sigma_path = os.path.join(sigma_path, file_)
    if os.path.exists(img_path) and os.path.exists(label_path):
        print('yes')
    else:
        print(file_) 

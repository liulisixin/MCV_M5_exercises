import GPUtil as gpu
gpu.assignGPU()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

from datasets import InShopDataset, Siamese_inshop, inshop_classes

import pickle

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit

from plot import extract_embeddings_high_dim, plot_embeddings, plot_history

experiment_folder = 'result_siamese_dim16'
batch_size = 64
embed_dim = 16

cuda = torch.cuda.is_available()

resume_from_pth = False
if not os.path.isdir(experiment_folder):
    os.makedirs(experiment_folder)
trained_weight_file = './{}/model.pth'.format(experiment_folder)

img_path = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Img'

img_file = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno/train_gallery_img.txt'
id_file = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno/train_gallery_id.txt'
train_dataset = InShopDataset(img_path, img_file, id_file, train=True)

img_file = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno/query_img.txt'
id_file = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno/query_id.txt'
test_dataset = InShopDataset(img_path, img_file, id_file, train=False)

# Set up data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
from metrics import AccumulatedAccuracyMetric

embedding_net = EmbeddingNet(embed_dim)
margin = 1.
model = SiameseNet(embedding_net)

if cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.cuda()

model.load_state_dict(torch.load(trained_weight_file))

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('log')

val_embeddings_baseline, val_labels_baseline, all_images = extract_embeddings_high_dim(test_loader, model, embed_dim)
writer.add_embedding(val_embeddings_baseline,
                    metadata=val_labels_baseline,
                    label_img=all_images)
writer.close()


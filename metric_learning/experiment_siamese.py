import GPUtil as gpu

gpu.assignGPU()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

from datasets import InShopDataset, Siamese_inshop

import pickle

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier

from trainer import fit

from plot import extract_embeddings, extract_embeddings_high_dim, plot_embeddings, plot_history

import argparse

# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--experiment_folder', type=str, default='result')
# parser.add_argument('--embed_dim', type=int, default=2)
# parser.add_argument('--batch_size', type=int, default=64)
# print('run with', parser.parse_args())
#
# experiment_folder = parser.parse_args().experiment_folder
# embed_dim = parser.parse_args().embed_dim
# batch_size = parser.parse_args().batch_size

experiment_folder = 'result_siamese_dim32'
batch_size = 64
embed_dim = 32

cuda = torch.cuda.is_available()

resume_from_pth = True
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

# gallery is used in train before, but we also need to plot it later.
# img_file = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno/gallery_img.txt'
# id_file = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno/gallery_id.txt'
# gallery_dataset = InShopDataset(img_path, img_file, id_file, train=False)

# Set up data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# gallery_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

siamese_train_dataset = Siamese_inshop(train_dataset)  # Returns pairs of images and target same/different
siamese_test_dataset = Siamese_inshop(test_dataset)

siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

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
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 40
log_interval = 10

if not resume_from_pth:
    print("begin fit")
    record_history = fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs,
                         cuda, log_interval)
    with open('./{}/record_history.pkl'.format(experiment_folder), 'wb') as pkl_file:
        pickle.dump(record_history, pkl_file)
        pkl_file.close()

    torch.save(model.state_dict(), trained_weight_file)
else:
    with open('./{}/record_history.pkl'.format(experiment_folder), 'rb') as pkl_file:
        record_history = pickle.load(pkl_file)
        pkl_file.close()

    model.load_state_dict(torch.load(trained_weight_file))

plot_history(experiment_folder, record_history)

if embed_dim > 2:
    train_embeddings_baseline, train_labels_baseline, train_all_images = extract_embeddings_high_dim(train_loader,
                                                                                                     model, embed_dim)
    val_embeddings_baseline, val_labels_baseline, val_all_images = extract_embeddings_high_dim(test_loader, model,
                                                                                               embed_dim)
else:
    train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
    val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
plot_embeddings(experiment_folder, 'train', train_embeddings_baseline, train_labels_baseline)
plot_embeddings(experiment_folder, 'test', val_embeddings_baseline, val_labels_baseline)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(train_embeddings_baseline, train_labels_baseline)

score = knn.score(val_embeddings_baseline, val_labels_baseline)
print(score)

pass

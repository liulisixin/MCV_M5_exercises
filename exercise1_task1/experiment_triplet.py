from datasets import InShopDataset, Triplet_inshop
import os
import pickle

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit

from plot import extract_embeddings, plot_embeddings, plot_history

import GPUtil as gpu
gpu.assignGPU()

cuda = torch.cuda.is_available()

# Baseline: Classification with softmax
experiment_folder = 'result_triplet'
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
batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

triplet_train_dataset = Triplet_inshop(train_dataset) # Returns triplets of images
triplet_test_dataset = Triplet_inshop(test_dataset)

triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss

embed_dim = 2
embedding_net = EmbeddingNet(embed_dim)
margin = 1.
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

if not resume_from_pth:
    print("begin fit")
    record_history = fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs,
                         cuda, log_interval)
    with open('./{}/record_history.pkl'.format(experiment_folder), 'wb') as pkl_file:
        pickle.dump(record_history, pkl_file)
        pkl_file.close()

    torch.save(model.state_dict(), trained_weight_file)
else:
    with open('./{}/record_history.pkl'.format(experiment_folder), 'rb') as pkl_file:
        record_history = pickle.load(pkl_file)
        pkl_file.close()
    plot_history(experiment_folder, record_history)

    model.load_state_dict(torch.load(trained_weight_file))

train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
plot_embeddings(experiment_folder, 'train', train_embeddings_baseline, train_labels_baseline)
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
plot_embeddings(experiment_folder, 'test', val_embeddings_baseline, val_labels_baseline)

pass


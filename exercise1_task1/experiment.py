from datasets import InShopDataset
import os

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
experiment_folder = 'result_Classification_with_softmax'
if not os.path.isdir(experiment_folder):
    os.makedirs(experiment_folder)
trained_weight_file = './{}/model.pth'.format(experiment_folder)

resume_from_pth = True

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

# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric

n_items = 7981

embed_dim = 2
embedding_net = EmbeddingNet(embed_dim)
model = ClassificationNet(embedding_net, n_classes=n_items)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
n_epochs = 20
log_interval = 1

if not resume_from_pth:
    record_history = fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
    plot_history(experiment_folder, record_history)

    torch.save(model.state_dict(), trained_weight_file)

model.load_state_dict(torch.load(trained_weight_file))

train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
plot_embeddings(experiment_folder, 'train', train_embeddings_baseline, train_labels_baseline)
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
plot_embeddings(experiment_folder, 'test', val_embeddings_baseline, val_labels_baseline)

pass


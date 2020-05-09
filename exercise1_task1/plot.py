import torch
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

inshop_classes = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 'Suiting', 'Sweaters',
                  'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Denim', 'Dresses',
                  'Graphic_Tees', 'Jackets_Coats', 'Leggings', 'Pants', 'Rompers_Jumpsuits',
                  'Shorts', 'Skirts', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks']

colors = plt.cm.get_cmap('hsv', len(inshop_classes))


def plot_history(experiment_folder, record_history):
    # record history is [train_loss, train_metric_value, val_loss, val_metric_value]
    plt.plot([x[1] for x in record_history])
    plt.plot([x[3] for x in record_history])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./{}/accuracy.jpg'.format(experiment_folder))
    plt.close()
    # summarize history for loss
    plt.plot([x[0] for x in record_history])
    plt.plot([x[2] for x in record_history])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./{}/loss.jpg'.format(experiment_folder))


def plot_embeddings(experiment_folder, name, embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(len(inshop_classes)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors(i))
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(inshop_classes)
    plt.savefig('./{}/embedding_{}.jpg'.format(experiment_folder, name))
    plt.close()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        categories = dataloader.dataset.categories
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    # replace labels with the categories id.
    for i in range(labels.shape[0]):
        category = categories[int(labels[i])]
        labels[i] = inshop_classes.index(category)

    return embeddings, labels



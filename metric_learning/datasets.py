import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms

# target_type = 'item_id'
target_type = 'garment_categories'
# inshop_classes = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 'Suiting', 'Sweaters',
#                   'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Denim', 'Dresses',
#                   'Graphic_Tees', 'Jackets_Coats', 'Leggings', 'Pants', 'Rompers_Jumpsuits',
#                   'Shorts', 'Skirts', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks']
# Men and Women have some identical catogories, delete them.
inshop_classes = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 'Suiting', 'Sweaters',
                  'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Dresses',
                  'Graphic_Tees', 'Jackets_Coats', 'Leggings', 'Rompers_Jumpsuits',
                  'Skirts']


class InShopDataset(Dataset):
    # InShopDataset
    def __init__(self,
                 img_path,
                 img_file,
                 id_file,
                 train=True,
                 ):
        self.train = train
        # img_size = (224, 224)
        # img_size = (28, 28)
        img_size = (256, 256)
        self.img_size = img_size

        self.img_path = img_path

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # read img names
        fp = open(img_file, 'r')
        self.img_list = [x.strip() for x in fp]

        # extract categories from the img_list.
        self.categories = [x.split('/')[2] for x in self.img_list]
        self.catogories_id = [inshop_classes.index(x) for x in self.categories]

        # collect id
        # id is item id, idx is the image id.
        self.ids = []
        id_fn = open(id_file).readlines()
        self.id2idx, self.idx2id = {}, {}
        for idx, line in enumerate(id_fn):
            img_id = int(line.strip('\n'))
            self.ids.append(img_id)
            self.idx2id[idx] = img_id

            if img_id not in self.id2idx:
                self.id2idx[img_id] = [idx]
            else:
                self.id2idx[img_id].append(idx)
        fp.close()

    def __getitem__(self, idx):
        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is index of the target class.
                """
        if target_type == 'item_id':
            target = self.ids[idx]
        else:
            target = self.catogories_id[idx]


            # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.img_path, self.img_list[idx]))
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_list)


class Siamese_inshop(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, inshop_dataset):
        self.inshop_dataset = inshop_dataset

        self.train = self.inshop_dataset.train
        self.transform = self.inshop_dataset.transform
        self.img_size = self.inshop_dataset.img_size

        self.img_path = self.inshop_dataset.img_path

        if target_type == 'item_id':
            self.labels = self.inshop_dataset.ids
        else:
            self.labels = self.inshop_dataset.catogories_id
        self.data = self.inshop_dataset.img_list
        self.labels_set = set(np.array(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            # 29 is seed
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i]]),
                               1]
                              for i in range(0, len(self.data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.data[index], self.labels[index]
            if target == 1 and len(self.label_to_indices[label1]) > 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                target = 0
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.data[siamese_index]
        else:
            img1 = self.data[self.test_pairs[index][0]]
            img2 = self.data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.open(os.path.join(self.img_path, img1))
        img1.thumbnail(self.img_size, Image.ANTIALIAS)
        img1 = img1.convert('RGB')

        img2 = Image.open(os.path.join(self.img_path, img2))
        img2.thumbnail(self.img_size, Image.ANTIALIAS)
        img2 = img2.convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.inshop_dataset)


class Triplet_inshop(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, inshop_dataset):
        self.inshop_dataset = inshop_dataset

        self.train = self.inshop_dataset.train
        self.transform = self.inshop_dataset.transform
        self.img_size = self.inshop_dataset.img_size

        self.img_path = self.inshop_dataset.img_path

        if target_type == 'item_id':
            self.labels = self.inshop_dataset.ids
        else:
            self.labels = self.inshop_dataset.catogories_id
        self.data = self.inshop_dataset.img_list
        self.labels_set = set(np.array(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            # 29 is seed
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.data[index], self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.data[positive_index]
            img3 = self.data[negative_index]
        else:
            img1 = self.data[self.test_triplets[index][0]]
            img2 = self.data[self.test_triplets[index][1]]
            img3 = self.data[self.test_triplets[index][2]]

        img1 = Image.open(os.path.join(self.img_path, img1))
        img1.thumbnail(self.img_size, Image.ANTIALIAS)
        img1 = img1.convert('RGB')

        img2 = Image.open(os.path.join(self.img_path, img2))
        img2.thumbnail(self.img_size, Image.ANTIALIAS)
        img2 = img2.convert('RGB')

        img3 = Image.open(os.path.join(self.img_path, img3))
        img3.thumbnail(self.img_size, Image.ANTIALIAS)
        img3 = img3.convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.inshop_dataset)


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = set(np.array(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(list(self.labels_set), self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

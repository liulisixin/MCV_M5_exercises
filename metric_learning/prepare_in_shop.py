import os

# PREFIX = 'In-shop/Anno'
PREFIX = '/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Anno'


def split_img():
    # fn = open('In-shop/Eval/list_eval_partition.txt').readlines()
    fn = open('/home/yixiong/exercise/In-shop_Clothes_Retrieval_Benchmark/Eval/list_eval_partition.txt').readlines()
    train_gallery = open(os.path.join(PREFIX, 'train_gallery_img.txt'), 'w')
    query = open(os.path.join(PREFIX, 'query_img.txt'), 'w')

    for i, line in enumerate(fn[2:]):
        aline = line.strip('\n').split()
        img, _, prefix = aline[0], aline[1], aline[2]
        if prefix == 'train' or prefix == 'gallery':
            train_gallery.write(img)
            train_gallery.write('\n')
        else:
            if prefix == 'query':
                query.write(img)
                query.write('\n')

    train_gallery.close()
    query.close()


def split_ids():
    id2label = dict()
    rf = open(os.path.join(PREFIX, 'list_item_inshop.txt')).readlines()
    for i, line in enumerate(rf[1:]):
        id2label[line.strip('\n')] = i

    def write_id(rf, wf):
        for i, line in enumerate(rf):
            id = line.strip('\n').split('/')[3]
            label = id2label[id]
            wf.write('%s\n' % str(label))
        wf.close()

    rf1 = open(os.path.join(PREFIX, 'train_gallery_img.txt')).readlines()
    rf2 = open(os.path.join(PREFIX, 'query_img.txt')).readlines()

    wf1 = open(os.path.join(PREFIX, 'train_gallery_id.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'query_id.txt'), 'w')

    write_id(rf1, wf1)
    write_id(rf2, wf2)


if __name__ == '__main__':
    split_img()
    split_ids()

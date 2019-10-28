import os
import random

def run(src_path):
    '''
    gen ImageSet/Main/train.txt, val.txt, test.txt
    :param src_path:
    :return:
    '''
    # cur_path = os.getcwd()
    cur_path = src_path
    trainval_percent = 0.8    # train:test=8:2
    train_percent = 0.7       # train / trainval = 0.7
    fdir = os.path.join(cur_path, 'ImageSets/Main/')
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if not os.path.exists(fdir):
        raise Exception("%s not exits"%(fdir))

    xmlfilepath = os.path.join(cur_path, 'Annotations')

    txtsavepath = fdir
    total_xml = os.listdir(xmlfilepath)

    num=len(total_xml)
    list=range(num)
    tv=int(num*trainval_percent)
    tr=int(tv*train_percent)
    trainval= random.sample(list,tv)
    train=random.sample(trainval,tr)

    ftrainval = open(fdir + 'trainval.txt', 'w')
    ftest = open(fdir + 'test.txt', 'w')
    ftrain = open(fdir + 'train.txt', 'w')
    fval = open(fdir + 'val.txt', 'w')

    for i  in list:
        name=total_xml[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()

if __name__=='__main__':
    run(src_path='.')
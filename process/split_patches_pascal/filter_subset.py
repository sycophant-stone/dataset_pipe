import os
import random
import utils.shell as shell
from random import randint

def fill_content_for_subset(src_subset_file, src_dataset_images_dir, src_dataset_annos_dir, dst_subset_image_dir, dst_subset_anno_dir):
    '''
    fill the subset with images and annotations according to subset file.
    :param src_subset_file:
    :param dst_subset_content:
    :return:
    '''

    with open(src_subset_file, 'r') as f:
        for line in f.readlines():
            words = line.strip()
            src_image = os.path.join(src_dataset_images_dir, words+".jpg")
            if not os.path.exists(src_image):
                raise Exception("%s not exists"%(src_image))
            src_anno = os.path.join(src_dataset_annos_dir, words+".xml")
            if not os.path.exists(src_anno):
                raise Exception("%s not exists"%(src_anno))
            cmd = "cp %s %s"%(src_image, dst_subset_image_dir)
            shell.run_system_command(cmd)
            cmd = "cp %s %s"%(src_anno, dst_subset_anno_dir)
            shell.run_system_command(cmd)


def run(src_origin_set_file, src_filter_ratio,dst_subset_file):
    '''
    random filer given set by ratio
    :param src_origin_set_file:
    :param src_filter_ratio:
    :param dst_subset_file:
    :return:
    '''

    with open(src_origin_set_file,'r') as f:
        len_origin_set = len(f.readlines())

    resultList = random.sample(range(0, len_origin_set), int(len_origin_set*src_filter_ratio))

    dstp = open(dst_subset_file, 'w')

    with open(src_origin_set_file,'r') as f:
        for index, line in enumerate(f.readlines()):
            if index in resultList:
                dstp.write(line)

    dstp.close()

if __name__=='__main__':
    run(src_origin_set_file='', src_filter_ratio=0.3, dst_subset_file='')
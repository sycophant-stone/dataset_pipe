#!/bin/bash
# baseon https://blog.csdn.net/Chris_zhangrx/article/details/80458515
# modified 
# working dir:
# FID_DID_HEAD_CLEAN_0_patches_int
#    JPEGImage
#    Annotations
#    ImageSets/Main
#    create_list.sh
#    create_data.sh
#
# Params:
# root_dir= path to parent of FID_DID_HEAD_CLEAN_0_patches_int
# sub_dir = fixed to ImageSets/Main
# `for name in ` name = FID_DID_HEAD_CLEAN_0_patches_int
# then $root_dir/$name can make the entair path.
#

# NOTES:
# the root dir 
#       I )   IS THE PARENT OF  includes the `Annotations` , `JPEGImages`, ImageSets`
#       II)   without the '/' at the end of the root_dir string.
# export PYTHONPATH with caffe's python path
# Examples:
# user@251aaaac4017:/ssd/hnren/Data/coco_1080p_head/FID_DID_HEAD_CLEAN_0$ ll
# total 2056
# drwxrwxrwx 5 root root   4096 Oct 14 06:53 ./
# drwxr-xr-x 6 root root    163 Oct 14 06:08 ../
# -rw-r--r-- 1 user user  12288 Oct 14 06:53 .create_data.sh.swp
# -rw-r--r-- 1 user user   4096 Oct 14 06:52 .create_list.sh.swp
# drwxrwxrwx 2 root root 352256 Oct 14 06:07 Annotations/
# drwxr-xr-x 3 user user     26 Oct 14 06:45 ImageSets/
# drwxrwxrwx 2 root root 352256 Oct 14 06:07 JPEGImages/
# -rwxrwxrwx 1 root root     72 Oct 14 06:35 auto_create_.sh*
# -rwxrwxrwx 1 root root   1033 Oct 14 06:45 cls.py*
# -rwxrwxrwx 1 root root   1268 Oct 14 06:53 create_data.sh*
# -rwxrwxrwx 1 root root   1991 Oct 14 06:50 create_list.sh*
# -rw-rw-rw- 1 root root 317062 Oct 14 06:07 name.txt
# -rwxrwxrwx 1 root root   1907 Oct 14 06:35 one_img_xml.py*
# -rw-r--r-- 1 root root 163860 Oct 14 06:51 test.txt
# -rw-r--r-- 1 root root  59140 Oct 14 06:52 test_name_size.txt
#

# root_dir=/ssd/hnren/Data/coco_1080p_head
root_dir=$1
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in $2
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    /ssd/hnren/tf/1sd/caffe/build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done

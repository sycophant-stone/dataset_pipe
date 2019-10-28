
# NOTES:
# the root dir 
#       I )   IS THE PARENT OF  includes the `Annotations` , `JPEGImages`, ImageSets`
#       II)   without the '/' at the end of the root_dir string.
# export PYTHONPATH with caffe's python path
#
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
# -rw-r--r-- 1 root root 655011 Oct 14 06:51 trainval.txt 
#


cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
# root_dir=/ssd/hnren/tf/1sd/caffe
root_dir=$1

cd $root_dir

redo=1
data_root_dir=$2
dataset_name=$3
mapfile=$4
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python $root_dir/scripts/create_annoset.py \
	  --anno-type=$anno_type \
	  --label-map-file=$mapfile \
	  --min-dim=$min_dim \
	  --max-dim=$max_dim \
	  --resize-width=$width \
	  --resize-height=$height \
	  --check-label $extra_cmd $data_root_dir \
	  $2/$3/$subset.txt \
	  $2/$3"_"$subset"_lmdb" $2/$3/examples
done

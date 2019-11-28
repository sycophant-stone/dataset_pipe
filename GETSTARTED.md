# setup

## dependency
 
pip install colored

pip install termcolor

## env setup

export PYTHONPATH=/ssd/hnren/tf/1sd/caffe/python/

> tips:\
  must note the `python/` with `/` as a end.



# how to

### gen_random_subset_pascal_dataset 


root@86d03b1c1aad:/ssd/hnren/Data/dataset_pipe# python pipe.py --yaml config/flow_config/gen_random_subset_pascal_dataset.yaml --data_dir /ssd/hnren/Data/dataset_pipe/newcrop --baseline_dataset_dir /ssd/hnren/Data/coco_300px_head/FID_DID_HEAD_CLEAN_0 --caffe_python_dir /ssd/hnren/tf/1sd/caffe


python pipe.py --yaml config/flow_config/gen_random_subset_pascal_dataset.yaml --data_dir /ssd/hnren/Data/dataset_pipe/tempout --baseline_dataset_dir /ssd/hnren/Data/coco_300px_head/FID_DID_HEAD_CLEAN_0 --caffe_python_dir /ssd/hnren/tf/1sd/caffe

### gen pascal format
python pipe.py --yaml config/flow_config/gen_pascal_format.yaml --data_dir /ssd/hnren/Data/dataset_pipe/tempout_patches_int_pascal --baseline_dataset_dir /ssd/hnren/Data/tempout_patches_int --caffe_python_dir /ssd/hnren/tf/1sd/caffe

### filter datasets
python pipe.py --yaml config/flow_config/filter_dataset.yaml --data_dir /ssd/hnren/Data/tempout_patches_int_pascal --baseline_dataset_dir /ssd/hnren/Data/tempout_patches_int --caffe_python_dir /ssd/hnren/tf/1sd/caffe


# others

## other docker for caffe

docker pull bvlc/caffe:gpu

nvidia-docker run -it -v /ssd:/ssd --name hhnnrr_caffe bvlc/caffe:gpu /bin/bash



## v2

python pipe.py --yaml config/flow_config/gen_random_subset_pascal_dataset.yaml --data_dir /ssd/hnren/Data/dataset_pipe/newcropv2 --baseline_dataset_dir /ssd/hnren/Data/coco_300px_head/FID_DID_HEAD_CLEAN_0 --caffe_python_dir /ssd/hnren/tf/1sd/caffe
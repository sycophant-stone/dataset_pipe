pip install colored
pip install termcolor



gen_random_subset_pascal_dataset.yaml


python pipe.py --yaml config/flow_config/gen_random_subset_pascal_dataset.yaml --data_dir /ssd/hnren/Data/dataset_pipe/tempout --baseline_dataset_dir /ssd/hnren/Data/coco_300px_head/FID_DID_HEAD_CLEAN_0 --caffe_python_dir /ssd/hnren/tf/1sd/caffe

python pipe.py --yaml config/flow_config/gen_pascal_format.yaml --data_dir /ssd/hnren/Data/dataset_pipe/tempout_patches_int_pascal --baseline_dataset_dir /ssd/hnren/Data/tempout_patches_int --caffe_python_dir /ssd/hnren/tf/1sd/caffe

python pipe.py --yaml config/flow_config/filter_dataset.yaml --data_dir /ssd/hnren/Data/tempout_patches_int_pascal --baseline_dataset_dir /ssd/hnren/Data/tempout_patches_int --caffe_python_dir /ssd/hnren/tf/1sd/caffe


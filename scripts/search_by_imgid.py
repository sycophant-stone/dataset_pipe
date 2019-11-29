import os,cv2
from process.split_patches_pascal.gen_size_dataset import *


def vis_bboxes_by_imgid(src_imgid, src_raw_map_file, src_crop_map_file, src_reclip_map_file, src_img_dir, dst_res_dir):
    '''
    vis all related bboxes by imgid.
    :param src_imgid:
    :param src_raw_map_file:
    :param src_crop_map_file:
    :param src_reclip_map_file:
    :param src_img_dir:
    :param dst_res_dir:
    :return:
    '''
    dst_res_dir_imgid = os.path.join(dst_res_dir,src_imgid)
    if not os.path.exists(dst_res_dir_imgid):
        os.mkdir(dst_res_dir_imgid)
    raw_map_file = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=src_raw_map_file)
    crop_map_file = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=src_crop_map_file)
    reclip_map_file = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=src_reclip_map_file)

    raw_bboxes_list = raw_map_file[src_imgid]
    imgpath = os.path.join(src_img_dir, src_imgid + ".jpg")
    imgpath_raw_save = os.path.join(dst_res_dir_imgid, src_imgid + "_with_rawbox.jpg")
    imgpath_crop_save = os.path.join(dst_res_dir_imgid, src_imgid + "_with_cropbox.jpg")
    imgpath_reclip_save = os.path.join(dst_res_dir_imgid, src_imgid + "_with_reclip.jpg")
    imgpath_save = os.path.join(dst_res_dir_imgid, src_imgid + "_with_all_bboxes.jpg")
    if not os.path.exists(imgpath):
        raise Exception('%s not exists!' % (imgpath))
    else:
        oimg = cv2.imread(imgpath)
    rawimg = oimg.copy()
    cropimg = oimg.copy()
    reclipimg = oimg.copy()
    allimg = oimg

    rawimg = visionalize_bboxes_list_on_img_dat(
        src_img_data=rawimg,
        src_bboxes_list=raw_bboxes_list,
        src_color_idx=3,
        src_color_width=2
    )
    allimg = visionalize_bboxes_list_on_img_dat(
        src_img_data=allimg,
        src_bboxes_list=raw_bboxes_list,
        src_color_idx=3,
        src_color_width=2
    )

    crop_bboxes_list = crop_map_file[src_imgid]
    cropimg = visionalize_bboxes_list_on_img_dat(
        src_img_data=cropimg,
        src_bboxes_list=crop_bboxes_list,
        src_color_idx=1,
        src_color_width=2
    )
    allimg = visionalize_bboxes_list_on_img_dat(
        src_img_data=allimg,
        src_bboxes_list=crop_bboxes_list,
        src_color_idx=1,
        src_color_width=2
    )

    reclip_bboxes_list = reclip_map_file[src_imgid]
    reclipimg = visionalize_bboxes_list_on_img_dat(
        src_img_data=reclipimg,
        src_bboxes_list=reclip_bboxes_list,
        src_color_idx=2,
        src_color_width=3
    )
    allimg = visionalize_bboxes_list_on_img_dat(
        src_img_data=allimg,
        src_bboxes_list=reclip_bboxes_list,
        src_color_idx=2,
        src_color_width=3
    )

    cv2.imwrite(imgpath_raw_save, rawimg)
    cv2.imwrite(imgpath_crop_save, cropimg)
    cv2.imwrite(imgpath_reclip_save, reclipimg)
    cv2.imwrite(imgpath_save, allimg)


if __name__ == '__main__':
    vis_bboxes_by_imgid(
        src_imgid='ch01001_20190318_ch01001_20190318110500.mp4.cut.mp4_003000',
        src_raw_map_file='/ssd/hnren/Data/dataset_pipe/newcropv3/raw_imgid_boxes_map.csv',
        src_crop_map_file='/ssd/hnren/Data/dataset_pipe/newcropv3/cropped_imgid_bboxes_map_file.csv',
        src_reclip_map_file='/ssd/hnren/Data/dataset_pipe/newcropv3/reclip_imgid_bboxes_map_file.csv',
        src_img_dir='/ssd/hnren/Data/dataset_pipe/newcropv3/JPEGImages',
        dst_res_dir='/ssd/hnren/Data/dataset_pipe/eva'
    )

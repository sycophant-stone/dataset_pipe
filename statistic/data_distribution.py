import os
import numpy as np
import utils.file as file
from dataset_lib.pascal_voc import PascalVocAnn

def GET_BARENAME(fullname):
    try:
        return os.path.splitext(os.path.basename(fullname))[0]
    except:
        raise Exception("%s os actions error "%(fullname))

def statistic_bbox_distribution(src_xml_dir,src_img_dir,src_sml_size_thresh):
    '''
    statistic bbox distributions.
    :param src_xml_dir:
    :param src_img_dir:
    :param src_sml_size_thresh:  rectangles's size.
    :return:
    '''
    imgs_list = file.list_all_files(src_img_dir, exts=["jpg"])
    xmls_list = file.list_all_files(src_xml_dir, exts=["xml"])

    bboxes_area_list = []
    imgname_area_map = {}
    for xml in xmls_list:
        pascal_voc_ann = PascalVocAnn(xml=lb_filepath)
        bboxes = pascal_voc_ann.get_boxes()
        h, w, c = pascal_voc_ann.get_size()
        imgname = GET_BARENAME(xml)
        area = float(h)*float(w)
        imgname_area_map[imgname] = area
        bboxes_area_list.append(area)

    np_bboxes_area = np.array(bboxes_area_list)

    bmax = np.max(np_bboxes_area)
    bmin = np.min(np_bboxes_area)
    blen = len(np_bboxes_area)
    histogram_bin =(bmax-bmin)/blen
    n,bin = np.histogram(np_bboxes_area,int(histogram_bin))
    print("max: %s, min: %s, len: %s"%(bmax, bmin , blen))
    print("histogram: %s"%(n))


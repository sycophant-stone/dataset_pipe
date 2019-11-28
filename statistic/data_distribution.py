import os
import cv2
import numpy as np
import utils.file as file
import operator
import utils.shell as shell
from utils.log_table import LogTable
from dataset_lib.pascal_voc import PascalVocAnn

def GET_BARENAME(fullname):
    try:
        return os.path.splitext(os.path.basename(fullname))[0]
    except:
        raise Exception("%s os actions error "%(fullname))

def gen_xml_hist_info_map(src_xml_dir, src_hist_bin_numbers):
    '''
    gen xml distributions and
    gen histogram's info and
    gen area xml map.
    :param src_xml_dir:
    :return:
        nlist: histogram distributions' counts
        bin_edge_list: each bin's edge(organizing them by edge_start and edge_end)
        np_bboxes_area: a list for saving all bouding boxes area, which has no relations with xml file.
        area_xml_map: a map. key is area, value is corresponding xml file,which is helpful for reference.
        area_bboxenum_map:  a map, key is the area, value is corresponding bboxes number.
                            Take it consider that the one xml may have more than one bboxes, and many xml' bboxes may share the same area.
    '''
    xmls_list = file.list_all_files(src_xml_dir, exts=["xml"])

    print("Target xml dir: %s, with len: %s"%(src_xml_dir, len(xmls_list)))

    bboxes_area_list = []
    area_bboxenum_map = {}
    area_xml_map={}
    for idx,xml in enumerate(xmls_list):
        pascal_voc_ann = PascalVocAnn(xml=xml)
        bboxes = pascal_voc_ann.get_boxes()
        for box in bboxes:
            box = [int(b) for b in box[1:]]
            h = box[3]-box[1]
            w = box[2]-box[0]
            area = float(h)*float(w)
            bboxes_area_list.append(area)
            if area not in area_xml_map:
                area_xml_map[area]=set([])
            area_xml_map[area].add(xml)
            if area not in area_bboxenum_map:
                tmp = 1
                area_bboxenum_map[area] = tmp
            else:
                tmp = area_bboxenum_map[area]
                area_bboxenum_map[area] = tmp+1

    np_bboxes_area = np.array(bboxes_area_list)

    bmax = np.max(np_bboxes_area)
    bmin = np.min(np_bboxes_area)
    blen = len(np_bboxes_area)
    hbinsize = (float(bmax)-float(bmin))/src_hist_bin_numbers
    print("bmax:%s, bmin:%s, blen:%s, hbin:%s, hbinsize:%s"%(bmax, bmin, blen, src_hist_bin_numbers, hbinsize))
    nlist,bin_edge_list = np.histogram(np_bboxes_area,int(src_hist_bin_numbers))
    print("-----------------------------------------------------------")
    print("Tips:  those info below is belong to bbox instead of xmls..")
    lt = LogTable(["area distributions", "bboxes counts"])
    for idx, n in enumerate(nlist):
        edgeS = int(bin_edge_list[idx])
        edgeE = int(bin_edge_list[idx+1])
        lt.add_line(["%s -> %s"%(edgeS, edgeE), "%s"%(n)])
    lt.show()

    return nlist,bin_edge_list,np_bboxes_area,area_xml_map,area_bboxenum_map

def get_xml_by_area(src_area_xml_map, area):
    if area in src_area_xml_map.keys():
        return src_area_xml_map[area]
    else:
        print("area of %s is Not in src_area_xml_map"%(area))


def gen_bin_xml_map(src_area_xml_map, src_bin_cnt_list, src_bin_edge_list,src_area_bboxenum_map):
    '''
    gen bin xml map
    :param src_area_xml_map:
    :param src_bin_cnt_list:
    :param src_bin_edge_list:
    :return:
        bin_xml_map:
        key: 930-5353:
        val: corresponding xmls
    '''
    # gen bin's start end map
    start_end_map={}
    for idx, cnt in enumerate(src_bin_cnt_list):
        edgeS = int(src_bin_edge_list[idx])
        edgeE = int(src_bin_edge_list[idx+1])
        start_end_map[edgeS] = edgeE
    # print("start_end_map: ", start_end_map)

    # the sorted list would be [(start0,end0), (start1,end1)...]
    sorted_start_end_list = sorted(start_end_map.items(), key=operator.itemgetter(1), reverse=False)
    # print("sorted_start_end_list: ", sorted_start_end_list)

    bin_xml_map = {}
    bin_bboxes_number_map = {}
    # debug_cnt=10
    for area,xmls in src_area_xml_map.items():
        for bin_slice in sorted_start_end_list:
            es = float(bin_slice[0])
            ee = float(bin_slice[1])
            # print("area:%s, es:%s, ee:%s"%(area, es, ee))
            if float(area) > ee:
                continue
            else:
                binkey = '{}-{}'.format(int(es),int(ee))
                if binkey not in bin_xml_map:
                    bin_xml_map[binkey]=set([])
                for xml in xmls:
                    bin_xml_map[binkey].add(xml)

                if binkey not in bin_bboxes_number_map:
                    bin_bboxes_number_map[binkey] = int(src_area_bboxenum_map[area])
                else:
                    tmp = bin_bboxes_number_map[binkey]
                    bin_bboxes_number_map[binkey] = tmp + int(src_area_bboxenum_map[area])
                break
            # print(bin_xml_map)
            # if debug_cnt==0:
            #     raise Exception("stop ..")
            # else:
            #     debug_cnt=debug_cnt-1
    # print(bin_xml_map)

    return bin_xml_map,bin_bboxes_number_map


def save_imgs_according_bin_xml_map(src_img_dir, src_bin_xml_map, dst_img_dir,src_bin_boxes_number_map):
    '''
    save imgs according bin xml map.
    root/
        edgestart0_edgeend0/
            1.jpg
            2.jpg
            77.jpg
            ...
        edgestart1_edgeend1/
            25.jpg
            ...

    :param src_img_dir:
    :param src_bin_xml_map:
    :param src_bin_boxes_number_map:
            a map. key is bin edge; value is corresponding bboxes number which belongs to this bin edge.
    :return:
    '''
    for edge_info,xmls in src_bin_xml_map.items():
        print("process.. %s with %s xmls , with %s bboxes"%(edge_info, len(xmls), src_bin_boxes_number_map[edge_info]))
        round_edge_dir = os.path.join(dst_img_dir, edge_info.replace('-','_'))
        if not os.path.exists(round_edge_dir):
            os.mkdir(round_edge_dir)
        for xml in xmls:
            barename=GET_BARENAME(xml)
            src_round_img_path = src_img_dir + os.sep + barename + '.jpg'
            if not os.path.exists(src_round_img_path):
                raise Exception("%s not exists!"%(src_round_img_path))
            cmd = 'cp %s %s'%(src_round_img_path, round_edge_dir)
            shell.run_system_command(cmd)
        # raise Exception("stop ..")

def visionalize_gt_with_distributions(src_img_dir, src_xml_dir):
    '''
    visionalize gt with distributions' folders.
    :param src_img_dir:
    :param src_xml_dir:
    :return:
    '''
    xmls_list = file.list_all_files(src_xml_dir, exts=["xml"])
    imgs_list = file.list_all_files(src_img_dir, exts=["jpg"])

    img_path_map = {}
    for imgpath in imgs_list:
        imgname = GET_BARENAME(imgpath)
        img_path_map[imgname] = imgpath

    for idx,xml in enumerate(xmls_list):
        pascal_voc_ann = PascalVocAnn(xml=xml)
        imgname = GET_BARENAME(xml)
        if imgname not in img_path_map.keys():
            # raise Exception("%s is Not in img_path_map"%(imgname))
            print("%s is Not in img_path_map"%(imgname))
            continue
        # img_path = os.path.join(img_path_map[imgname], imgname)
        img_path = img_path_map[imgname]
        if not os.path.exists(img_path):
            raise Exception("%s not exists!"%(img_path))
        else:
            img = cv2.imread(img_path)
        bboxes = pascal_voc_ann.get_boxes()
        for box in bboxes:
            box = [int(b) for b in box[1:]]
            color = (255, 0, 0)
            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.imwrite(img_path,img)


def statistic_bbox_distribution(src_xml_dir,src_img_dir,src_sml_size_thresh, src_hist_bin_num, dst_img_dir):
    '''
    statistic bbox distributions.
    :param src_xml_dir:
    :param src_img_dir:
    :param src_sml_size_thresh:  rectangles's size.
    :return:
    '''

    print('running.. gen xml hist info map')

    nlist,\
    bin_edge_list,\
    np_bboxes_area,\
    area_xml_map,\
    area_bboxenum_map \
        = gen_xml_hist_info_map(
        src_xml_dir=src_xml_dir,
        src_hist_bin_numbers=int(src_hist_bin_num)
    )

    bmax = np.max(np_bboxes_area)
    bmin = np.min(np_bboxes_area)
    blen = len(np_bboxes_area)
    print("bmax:%s, bmin:%s, blen:%s, hbin:%s"%(bmax, bmin, blen, src_hist_bin_num))

    min_xmls = get_xml_by_area(area_xml_map,bmin)
    for idx,mx in enumerate(min_xmls):
        print("%s/%s for bmin xml: %s"%(idx+1, len(min_xmls), mx))

    max_xmls = get_xml_by_area(area_xml_map,bmax)
    for idx, mx in enumerate(max_xmls):
        print("%s/%s for bmax xml: %s" % (idx+1, len(max_xmls), mx))

    print("running.. gen bin xml map")
    bin_xml_map, bin_bboxes_number_map = gen_bin_xml_map(
        src_area_xml_map=area_xml_map,
        src_bin_cnt_list=nlist,
        src_bin_edge_list=bin_edge_list,
        src_area_bboxenum_map=area_bboxenum_map
    )
    print("running.. saving imgs by bin xml map")
    save_imgs_according_bin_xml_map(
        src_img_dir=src_img_dir,
        src_bin_xml_map=bin_xml_map,
        dst_img_dir=dst_img_dir,
        src_bin_boxes_number_map= bin_bboxes_number_map
    )
    print("running.. visionalize gt bbox , in distributions folders.")
    visionalize_gt_with_distributions(
        src_img_dir= dst_img_dir,
        src_xml_dir= src_xml_dir
    )

def Test_np_where():
    np_bboxes_area = np.array([1,2,3,4,55,66,1,77,88,999])
    print("np_bboxes_area: ", np_bboxes_area)
    inds=np.squeeze(np.where(1==np_bboxes_area))
    inds=list(inds)
    print("inds: ", inds)
    for ind in inds:
        print(ind)
        print(np_bboxes_area[ind])


if __name__=='__main__':
    Test_np_where()

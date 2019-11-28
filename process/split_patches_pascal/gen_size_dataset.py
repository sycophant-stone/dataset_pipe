import os, sys
import cv2, shutil
import numpy as np
import numpy.random as npr
import random
import math
import json
from utils.file import GET_BARENAME
from dataset_lib.pascal_voc import PascalVocAnn
from utils.bbox_file import get_imgid_bboxes_map_from_file
from utils.visionalize import visionalize_bboxes_list_on_img_dat
import utils.visionalize as visionalize


# from myAffine import *
# from drawShape import *

def distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def get_bbox_from_ldmk(pts):
    nose = [(pts[47 * 2] + pts[56 * 2]) / 2, (pts[47 * 2 + 1] + pts[56 * 2 + 1]) / 2]  # nose top
    chin = [pts[6 * 2], pts[6 * 2 + 1]]  # chin
    alpha = math.atan2(nose[1] - chin[1], nose[0] - chin[0]) + (3.14159 / 2)

    # affine warping the face
    srcCenter = np.array([(nose[0] + chin[0]) / 2.0, (nose[1] + chin[1]) / 2.0], np.float32)
    dstCenter = np.array([200, 200], np.float32)

    scale = 1
    warp_mat = Get_Affine_matrix(dstCenter, srcCenter, alpha, scale)

    min_x = 100000
    min_y = 100000
    max_x = -100000
    max_y = -100000
    for n in range(len(pts) / 2):
        srcpt = np.array([pts[2 * n], pts[2 * n + 1]], np.float32)
        dstpt = np.array([0, 0], np.float32)
        if srcpt[0] != -1 and srcpt[1] != -1:
            Affine_Point(warp_mat, srcpt, dstpt)
            if min_x > dstpt[0]:
                min_x = dstpt[0]
            if min_y > dstpt[1]:
                min_y = dstpt[1]
            if max_x < dstpt[0]:
                max_x = dstpt[0]
            if max_y < dstpt[1]:
                max_y = dstpt[1]
        else:
            dstpt[0] = -1
            dstpt[1] = -1

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    fw = max_x - min_x + 1
    fh = max_y - min_y + 1
    fsize = max(fh, fw)

    # adjust face center for profile faces
    leftpt = [pts[0], pts[1]]  # left profile
    rightpt = [pts[12 * 2], pts[12 * 2 + 1]]
    left_dist = 0
    right_dist = 0
    if (pts[13 * 2] >= 0 and pts[13 * 2 + 1] >= 0):
        left_dist = distance(leftpt, nose)
    if (pts[34 * 2] >= 0 and pts[34 * 2 + 1] >= 0):
        right_dist = distance(rightpt, nose)

    cx += (right_dist - left_dist) / (right_dist + left_dist) * fsize * 0.25

    # Transform (cx, cy) back to the original image
    inv_warp_mat = inverseMatrix(warp_mat)
    srcpt = np.array([cx, cy], np.float32)
    dstpt = np.array([0, 0], np.float32)
    Affine_Point(inv_warp_mat, srcpt, dstpt)
    cx = dstpt[0]
    cy = dstpt[1]

    return (cx, cy, fsize, alpha)


def mkdir(dr):
    if not os.path.exists(dr):
        os.makedirs(dr)


def read_all(file_names):
    lines = []
    for file_name in file_names:
        with open(file_name, 'rt') as f:
            lines.extend(f.readlines())
    return lines


# list all image files
def list_all_files(dir_name, exts=['jpg', 'bmp', 'png', 'xml']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result


def concat_files(in_filenames, out_filename):
    with open(out_filename, 'w') as outfile:
        for fname in in_filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def get_rus_by_head_box(box, headrawsz, head_out_base_sz=20, scale=0.2, outputsize=72):
    """
    param:
            box:  list, [xc, yc, sz]
            head_out_base_sz: head box output baseline size
            scale:  head box output percentage, aka, random
            outputsize: the whole size output
    return:
            need_crop_size:
                origin (big) img 's head need output size
    """
    new_size = (1 + random.uniform(-0.2, 0.2)) * head_out_base_sz
    # print(new_size)
    need_crop_size = headrawsz * outputsize / new_size
    # print("headrawsz:%f, head_out_base_sz:%f, outputsize:%f, new_size:%f, need_crop_size:%f"%(headrawsz, head_out_base_sz, outputsize, new_size, need_crop_size))
    return need_crop_size


def compute_IoU(box=None, gt_box=None):
    """
    Calculate the IoU value between two boxes.
    Args:
        box: box location information [xmin, ymin, xmax, ymax]
        gt_box: another box location information [xmin, ymin, xmax, ymax]
    Return:
        IoU Value, float.
    """
    # print("candinate box",box)
    # print("main box",gt_box)
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area_gt_box = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_x1 = max(box[0], gt_box[0])
    inter_y1 = max(box[1], gt_box[1])
    inter_x2 = min(box[2], gt_box[2])
    inter_y2 = min(box[3], gt_box[3])
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    area_inter = inter_w * inter_h
    ovr = float(area_inter) / float(area_box + area_gt_box - area_inter)
    return ovr, inter_x1, inter_y1, inter_x2, inter_y2


def get_related_bboxes(candinates, main_box):
    '''
    filter related bboxes
    candinates are the original(raw) picture's bboxes.
    main_box is the expanded bbox's img
    if there is some bboxes which are belong to the expanded bbox region. we
    had better label them in expanded bbox's img.
    and more, for some little head(face), we had better mask them rather than label them or unlabeled them.
    masking not unlabeling won't get the model confused,cause the negtive region is clean, not get polluted.
    :param candinates: the original img's all bboxes.
    :param main_box: the expanded box' img (or the cropped img)'s region.
    :return:
    need_psotprocess_boxes:
        [0,1,2,3] ==> bboxes
        [4] ==> whether this (0,1,2,3)'s region will get masked with gray color, cause for small patch GT boxes, which
                are inpropered to considered as positives with below 30 pixels width or heights.
                To avoid disterbutions, we had better mask them with gray color, aka, using Negtive backgroud replace those
                'so-called' Positive samples.
    '''
    need_psotprocess_boxes = []
    for (i, candi_box) in candinates.items():
        iou, xmin, ymin, xmax, ymax = compute_IoU(candi_box[0:4],
                                                  main_box)  # whether original gtboxes overlap with the img region.
        # print("iou",iou)
        if iou > 0:
            if (xmax - xmin) <= 30 or (ymax - ymin) <= 30:
                need_psotprocess_boxes.append([xmin, ymin, xmax, ymax, 1])
            else:
                need_psotprocess_boxes.append([xmin, ymin, xmax, ymax, 0])
    return need_psotprocess_boxes


def a_belong_b(a, b):
    '''
    bbox A belongs to B.
    means:
        B_xmin<A_xmin
        B_ymin<A_ymin
        A_xmax<B_xmax
        A_ymax<B_ymax
    :param a:
    :param b:
    :return:
    '''
    if a[0] > b[0] and a[1] > b[1] and a[2] < b[2] and a[3] < b[3]:
        return True
    else:
        return False


def get_related_bboxes_with_input_list(candinates, main_box):
    '''
    filter related bboxes
    candinates are the original(raw) picture's bboxes.
    main_box is the expanded bbox's img
    if there is some bboxes which are belong to the expanded bbox region. we
    had better label them in expanded bbox's img.
    and more, for some little head(face), we had better mask them rather than label them or unlabeled them.
    masking not unlabeling won't get the model confused,cause the negtive region is clean, not get polluted.
    :param candinates: the original img's all bboxes.
    :param main_box: the expanded box' img (or the cropped img)'s region.
    :return:
    need_psotprocess_boxes:
        [0,1,2,3] ==> bboxes
        [4] ==> whether this (0,1,2,3)'s region will get masked with gray color, cause for small patch GT boxes, which
                are inpropered to considered as positives with below 30 pixels width or heights.
                To avoid disterbutions, we had better mask them with gray color, aka, using Negtive backgroud replace those
                'so-called' Positive samples.
    '''
    need_psotprocess_boxes = []
    for i, candi_box in enumerate(candinates):
        # print("candi_box [%s, %s, %s, %s] with main_box [%s, %s, %s, %s]" % (
        #     candi_box[0], candi_box[1], candi_box[2], candi_box[3], main_box[0],
        #     main_box[1], main_box[2], main_box[3]))
        iou, xmin, ymin, xmax, ymax = compute_IoU(candi_box[0:4],
                                                  main_box)  # whether original gtboxes overlap with the img region.
        if iou > 0:
            if ((xmax - xmin) <= 30 or (ymax - ymin) <= 30) and not a_belong_b(a=candi_box, b=main_box):
                # print("too small region.. ignore.. iou:%s with [%s, %s, %s, %s] with diffx: %s, diffy: %s" % (
                #     iou, xmin, ymin, xmax, ymax, (xmax - xmin), (ymax - ymin)))
                need_psotprocess_boxes.append([xmin, ymin, xmax, ymax, 1])
            else:
                # print("satisfied region.. take it.. iou:%s with [%s, %s, %s, %s] with diffx: %s, diffy: %s" % (
                #     iou, xmin, ymin, xmax, ymax, (xmax - xmin), (ymax - ymin)))
                need_psotprocess_boxes.append([xmin, ymin, xmax, ymax, 0])
    # print("==============================")
    return need_psotprocess_boxes


def xytoxcyc(x, y, sz):
    '''
    convert x,y to xc, yc
    :param x:
    :param y:
    :param sz:
    :return:
    '''
    xc = x + (sz / 2)
    yc = y + (sz / 2)
    return xc, yc


def xcyctoxy(xc, yc, expand_sz, width, height):
    '''
    xc,yc to x,y
    :param xc:
    :param yc:
    :param expand_sz:
    :param width:
    :param height:
    :return:
    '''
    xmin = xc - expand_sz / 2
    ymin = yc - expand_sz / 2
    xmax = xc + expand_sz / 2
    ymax = yc + expand_sz / 2

    xmin = np.clip(xmin, 0, width - 1)
    xmax = np.clip(xmax, 0, width - 1)
    ymin = np.clip(ymin, 0, height - 1)
    ymax = np.clip(ymax, 0, height - 1)
    if xmin == 0:
        tmp_w = xmax - xmin
        xmax = expand_sz - tmp_w + xmax
    if xmax == width - 1:
        tmp_w = xmax - xmin
        xmin = xmin - (expand_sz - tmp_w)
    if ymin == 0:
        tmp_h = ymax - ymin
        ymax = expand_sz - tmp_h + ymax
    if ymax == height - 1:
        tmp_h = ymax - ymin
        ymin = ymin - (expand_sz - tmp_h)

    return xmin, ymin, xmax, ymax


def crop_and_gen_pascal(img_path, ori_xml, newvocdir, img_output_size, src_refine_rectangle_size=0.62,
                        gen_gt_rect=False):
    '''
    crop image to patched image.
    :param img_path: the target image
    :param ori_xml: the original image's xml
    :param newvocdir: the output fonder
    :param img_output_size: the required output image patch's size.
    :param gen_gt_rect: whether draw gt box on image and save those drawed images to output fonders.
    :return:
    '''

    # gen fonders
    if not os.path.exists(newvocdir):
        os.mkdir(newvocdir)
    njpegpth = os.path.join(newvocdir, 'JPEGImages')
    if not os.path.exists(njpegpth):
        os.mkdir(njpegpth)
    nannopth = os.path.join(newvocdir, 'Annotations')
    if not os.path.exists(nannopth):
        os.mkdir(nannopth)
    if gen_gt_rect:
        rectpth = os.path.join(newvocdir, 'RECT_JPEGImages')
        if not os.path.exists(rectpth):
            os.mkdir(rectpth)

    img_filepath = img_path
    lb_filepath = ori_xml
    try:
        img = cv2.imread(img_filepath)
    except:
        if not os.path.exists(img_filepath):
            raise Exception("%s not exists" % (img_filepath))

    pascal_voc_ann = PascalVocAnn(xml=lb_filepath)
    bboxes = pascal_voc_ann.get_boxes()
    h, w, c = pascal_voc_ann.get_size()
    # img_output_size = 72
    rectangle_boxes = {}
    rect_idx = 0
    # print("raw bbox",bboxes)
    for (i, b) in enumerate(bboxes):
        xmin, ymin, xmax, ymax = b[1:5]
        # print("raw xmin,ymin:(%d,%d), xmax,ymax:(%d,%d)"%(xmin,ymin,xmax,ymax))
        w_raw = xmax - xmin + 1
        h_raw = ymax - ymin + 1
        # sz = int(max(w_raw, h_raw))#  * 0.62)
        sz = int(max(w_raw, h_raw) * float(src_refine_rectangle_size))  # * 0.62)
        x = int(xmin + (w_raw - sz) * 0.5)
        y = int(ymin + h_raw - sz)
        new_xmin = x
        new_ymin = y
        new_xmax = x + sz - 1
        new_ymax = y + sz - 1
        # print("x,y,sz:(%d,%d,%d)",x,y,sz)
        new_xmin = new_xmin if new_xmin >= 0 else 0
        new_ymin = new_ymin if new_ymin >= 0 else 0
        new_xmax = new_xmax if new_xmax < w else w - 1
        new_ymax = new_ymax if new_ymax < h else h - 1

        # print("new xmin,ymin:(%d,%d), xmax,ymax:(%d,%d)"%(new_xmin,new_ymin,new_xmax,new_ymax))
        rectangle_boxes[i] = [new_xmin, new_ymin, new_xmax, new_ymax, sz]
        rect_idx = i

    for (i, boxlist) in rectangle_boxes.items():
        [xmin, ymin, xmax, ymax, sz] = boxlist
        # if i %1000==0:
        #     print("gen img %s/%s"%(i/1000, len(rectangle_boxes)/1000))
        xc = xmin + sz / 2
        yc = ymin + sz / 2
        # print('get crop info ...')
        # print("xmin,ymin:(%d,%d), xmax,ymax:(%d,%d)"%(xmin,ymin,xmax,ymax))
        rus = get_rus_by_head_box(box=[xc, yc, sz], headrawsz=sz, head_out_base_sz=20, scale=0.2, outputsize=72)
        rus = int(rus)
        rus = rus / 2
        # print("xc:%d, yc:%d, xmin:%d, ymin:%d, rus:%d"%(xc,yc,xmin,ymin,rus))
        # img_xmn = xc - rus if xc - rus >= 0 else 0
        # img_ymn = yc - rus if yc - rus >= 0 else 0
        # img_xmx = xc + rus if xc + rus < img.shape[1] else img.shape[1] - 1
        # img_ymx = yc + rus if yc - rus < img.shape[0] else img.shape[0] - 1
        # rectangle_boxes[rect_idx+1] = [img_xmn, img_ymn, img_xmx, img_ymx]
        # expand box to `img_output_size`
        # print("xc,yc:(%s,%s) sz:%s"%(xc,yc,sz))
        img_xmn, img_ymn, img_xmx, img_ymx = xcyctoxy(xc, yc, img_output_size, img.shape[1], img.shape[0])
        # print("after (%s,%s) and (%s,%s)"%(img_xmn,img_ymn,img_xmx,img_ymx))

        # filte out the crossed(or overlaped original bboxes with those expanded img region)
        # more, will record the masked region.which are too small head for positive, and we won't like
        # them to be negtive either to confuse the model's training.
        need_box = get_related_bboxes(rectangle_boxes, [img_xmn, img_ymn, img_xmx, img_ymx])
        tmp_img = img.copy()
        for nbox in need_box:
            if nbox[4] == 1:
                nbxmin, nbymin, nbxmax, nbymax = nbox[0:4]
                tmp_img[nbymin:nbymax, nbxmin:nbxmax] = 125
        ioregion_croped = tmp_img[img_ymn:img_ymx, img_xmn:img_xmx]
        w_croped = ioregion_croped.shape[1]
        h_croped = ioregion_croped.shape[0]
        # print('resize img from %sx%s to %sx%s ...' % (rus, rus, img_output_size, img_output_size))
        # ioregion = cv2.resize(ioregion_croped, (img_output_size, img_output_size), interpolation=cv2.INTER_CUBIC)
        # print('... done')
        ioregion = ioregion_croped

        new_anns_folder = os.path.join(newvocdir, "Annotations")
        new_imgs_folder = os.path.join(newvocdir, "JPEGImages")
        crop_img_name = os.path.splitext(img_filepath)[0] + "_crop_%d.jpg" % (i)
        crop_img_savepath = new_imgs_folder + "/" + os.path.basename(crop_img_name)
        cv2.imwrite(crop_img_savepath, ioregion)

        # print('rebase ori bbox pos ...')
        rebased_need_box = []
        # print("need_box",need_box)
        for boxlist in need_box:
            if boxlist[4] == 0:
                [box_xmin, box_ymin, box_xmax, box_ymax] = boxlist[0:4]
                reb_xmin = box_xmin - img_xmn
                reb_ymin = box_ymin - img_ymn
                reb_xmax = box_xmax - img_xmn
                reb_ymax = box_ymax - img_ymn
                reb_xmin = int(float(img_output_size) / w_croped * reb_xmin)
                reb_ymin = int(float(img_output_size) / h_croped * reb_ymin)
                reb_xmax = int(float(img_output_size) / w_croped * reb_xmax)
                reb_ymax = int(float(img_output_size) / h_croped * reb_ymax)
                rebased_need_box.append([reb_xmin, reb_ymin, reb_xmax, reb_ymax])

        # print('... done')
        # print('gen new xml ...')
        if gen_gt_rect:
            print('draw rectangle ...')
            region_rect = cv2.rectangle(ioregion, (int(nxmin), int(nymin)), (int(nxmax), int(nymax)), (0, 255, 0), 2)
            rect_imgs_folder = os.path.join(newvocdir, "RECT_JPEGImages")
            rect_img_name = os.path.splitext(img_filepath)[0] + "_rect_%d.jpg" % (i)
            rect_img_savepath = rect_imgs_folder + "/" + os.path.basename(rect_img_name)
            cv2.imwrite(rect_img_savepath, region_rect)

        crop_xml_name = os.path.splitext(os.path.basename(crop_img_savepath))[0] + ".xml"
        crop_xml_savepath = new_anns_folder + "/" + crop_xml_name
        newpascal_ann = PascalVocAnn(image_name=crop_img_savepath)
        newpascal_ann.set_filename(file_name=crop_img_savepath)
        newpascal_ann.set_size(size=[img_output_size, img_output_size, img.shape[2]])
        for reb_box in rebased_need_box:
            [reb_xmin, reb_ymin, reb_xmax, reb_ymax] = reb_box
            newpascal_ann.add_object(object_class="head", xmin=reb_xmin, ymin=reb_ymin, xmax=reb_xmax, ymax=reb_ymax)
        newpascal_ann.check_boxes()
        newpascal_ann.write_xml(crop_xml_savepath)
        # print('... done')


def gen_square_from_rectangle(src_bboxes, src_refine_rectangle_size, src_boundrary):
    '''
    gen square from rectangels.
    Tips:
        bboxes belongs to img.
    :param src_bboxes: rectangles' bboxes positions.
    :param src_refine_rectangle_size:
    :param src_boundrary: imgs' boundary list. [h, w]
    :return:
    '''
    rxmin, rymin, rxmax, rymax = src_bboxes
    rw = rxmax - rxmin + 1
    rh = rymax - rymin + 1

    square_size = int(max(rw, rh) * float(src_refine_rectangle_size))
    sxmin = int(rxmin + (rw - square_size) * 0.5)
    symin = int(rymin + (rh - square_size) * 0.5)
    sxmax = sxmin + square_size - 1
    symax = symin + square_size - 1

    img_h, img_w = src_boundrary
    sxmin = sxmin if sxmin > 0 else 0
    symin = symin if symin > 0 else 0
    sxmax = sxmax if sxmax < img_w else img_w - 1
    symax = symax if symax < img_h else img_h - 1

    return [sxmin, symin, sxmax, symax]


def Test_gen_square_from_rectangle():
    '''
    Test for gen_square_from_rectangle
    :return:
    '''
    xml_path = "/ssd/hnren/Data/dataset_pipe/newcrop/Annotations/ch00005_20190214_ch00005_20190214115052.mp4.cut.mp4_003000.xml"
    src_img_path = '/ssd/hnren/Data/dataset_pipe/newcrop/JPEGImages'
    dst_test_img_dir = '.'

    imgname = GET_BARENAME(xml_path) + '.jpg'
    imgpath = os.path.join(src_img_path, imgname)
    if not os.path.exists(imgpath):
        raise Exception("%s not exists!" % (imgpath))
    else:
        img = cv2.imread(imgpath)
        dst_img_path = os.path.join(dst_test_img_dir, imgname)
    pascal_voc_ann = PascalVocAnn(xml=xml_path)
    bboxes = pascal_voc_ann.get_boxes()
    h, w, c = pascal_voc_ann.get_size()
    img_shape = [h, w]
    val_rs = [0, 85, 170, 255]
    val_gs = [85, 170, 255, 0]
    val_bs = [170, 255, 0, 85]

    for bbox in bboxes:
        bbox = bbox[1:5]
        color = (val_rs[0], val_gs[0], val_bs[0])
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        for idx, rect_ration in enumerate([0.65, 0.7, 0.75]):
            color = (val_rs[idx + 1], val_gs[idx + 1], val_bs[idx + 1])
            square_shape = gen_square_from_rectangle(src_bboxes=bbox,
                                                     src_refine_rectangle_size=rect_ration,
                                                     src_boundrary=img_shape)
            img = cv2.rectangle(img, (int(square_shape[0]), int(square_shape[1])),
                                (int(square_shape[2]), int(square_shape[3])), color, 2)

    cv2.imwrite(dst_img_path, img)

    print("bbox: ", bbox)
    print("img_shape: ", img_shape)
    print("square_shape: ", square_shape)


def gen_expand_shape(src_img_out_shape_list, src_bboxes_scale_shape, src_box_size_shape):
    '''
    gen expact expand shape size.
    :param src_center_list:
    :param src_img_out_shape_list:
    :return:
    '''
    out_img_width, out_img_height = src_img_out_shape_list
    out_box_width, out_box_height = src_bboxes_scale_shape
    cur_box_width, cur_box_height = src_box_size_shape

    expand_img_width = float(out_img_width) / float(out_box_width) * float(cur_box_width)
    expand_img_height = float(out_img_height) / float(out_box_height) * float(cur_box_height)

    return expand_img_width, expand_img_height


def gen_xcyctoxy(src_center_list, src_expand_size, src_img_shape):
    '''
    gen xy from xcenter, ycenter and corresponding width, height.
    :param src_center_list: [xcenter, ycenter]
    :param src_expand_size: [expand_width, expand_height]
    :param src_img_shape: [img_width, img_height]
    :return:
    '''
    xcenter, ycenter = src_center_list
    expand_width, expand_height = src_expand_size
    img_width, img_height = src_img_shape
    xmin = xcenter - expand_width / 2 if xcenter - expand_width / 2 > 0 else 0
    ymin = ycenter - expand_height / 2 if ycenter - expand_height / 2 > 0 else 0
    xmax = xcenter + expand_width / 2 if xcenter + expand_width / 2 < img_width else img_width - 1
    ymax = ycenter + expand_height / 2 if ycenter + expand_height / 2 < img_height else img_height - 1

    deltax = 0
    deltay = 0
    # we suppose the crop region not bigger than img region..
    if xmin == 0:
        deltax = expand_width / 2 - xcenter
        if (xmax + deltax < img_width):
            xmax = xmax + deltax
        else:
            raise Exception("xmax(%s) + deltax(%s) exceed the img right boundary(%s)" % (xmax, deltax, img_width))
    elif xmax == img_width - 1:
        deltax = expand_width / 2 - (img_width - 1 - xcenter)
        if (xmin - deltax) > 0:
            xmin = xmin - deltax
        else:
            raise Exception("xmin(%s) - deltax(%s) exceed the img left boundary(0)" % (xmin, deltax))

    if ymin == 0:
        deltay = expand_height / 2 - ycenter
        if (ymax + deltay < img_height):
            ymax = ymax + deltay
        else:
            raise Exception("ymax(%s) + deltay(%s) exceed the img bottom boundary(%s)" % (ymax, deltay, img_height))
    if ymax == img_height - 1:
        deltay = expand_height / 2 - (img_height - 1 - ycenter)
        if (ymin - deltay) > 0:
            ymin = ymin - deltay
        else:
            raise Exception("ymin(%s) - deltay(%s) exceed the img top boundary(0)" % (ymin, deltay))

    return [xmin, ymin, xmax, ymax]


def post_process_mask_bboxes(src_img, src_need_post_bboxes):
    '''
    mask not-inter and belongs to need post(need mask) bboxe's corresponding img region.
      i) get the inter region.
     ii) save img's inter region to temp.
    iii) fill this need mask bbox region with gray color(125).
     iv) restore the inter region with step ii) 's temp.
     Tips:  inter region  `belongs to` mask bbox  when inter region exists.

    :param src_img:
    :param src_need_post_bboxes:
    :return: post processed img.
    '''

    need_mask_bboxes_list = []
    normal_bboxes_list = []
    for bbox in src_need_post_bboxes:
        if bbox[4] == 1:
            need_mask_bboxes_list.append(bbox)
        elif bbox[4] == 0:
            normal_bboxes_list.append(bbox)
        else:
            raise Exception("invalid [4] of src_need_post_bboxes")

    for maskbox in need_mask_bboxes_list:
        for normalbox in normal_bboxes_list:
            iou, ixmin, iymin, ixmax, iymax = compute_IoU(normalbox, maskbox)
            if iou == 0:
                continue
            tmp_region = src_img[iymin:iymax, ixmin:ixmax].copy()
            src_img[maskbox[1]:maskbox[3], maskbox[0]:maskbox[2]] = 125
            src_img[iymin:iymax, ixmin:ixmax] = tmp_region

    return src_img


def gen_slice_patches_in_pascal_format(
        src_img_path,
        src_xml_path,
        dst_img_dir,
        dst_xml_dir,
        src_img_output_size,
        src_refine_rectangle_size,
        src_bboxes_type,
        src_bboxes_scale_shape,
        dst_raw_imgid_bboxes_map_file,
        dst_cropped_imgid_bboxes_map_file,
        dst_reclip_imgid_bboxes_map_file
):
    '''
    gen slice patches from origin xml & imgs within pascal format outputing.
    :param src_img_path:
    :param src_xml_path:
    :param dst_out_dir:
    :param src_img_output_size:
    :param src_refine_rectangle_size:
    :param src_gen_gt_rect:
    :param src_bboxes_type:
            whether the bboxes are square or not.(default bboxes are rectangle).
    :param src_bboxes_scale_shape:
            wheher the bboxes need the resize.
                if null which means no need for resizing.
                if not null, which means we need cv2.resize to get the satisfied shape.
    :param dst_raw_imgid_bboxes_map_file:
            the bboxes from the origin jpg(aka the 1080p or more resolution jpg).
    :param dst_cropped_imgid_bboxes_map_file:
            to get the small set of pictures, or to satisfy the required img output size(300,300), we design a region for cropping.
    :param dst_reclip_imgid_bboxes_map_file:
            in the above cropped imgs, which is descriped at `dst_cropped_imgid_bboxes_map_file`, we reclip those original bboxes belong to objects.
    :return:
    '''
    src_img_list = list_all_files(src_img_path, exts=["jpg"])
    raw_imgid_bboxes_map = {}  # imgid,b0xmin b0ymin b0xmax b0ymax,b1 b1 b1 b1,...
    cropped_imgid_bboxes_map = {}  # not the bbox, but the cropped rectangle region for statifying output img size.
    reclip_imgid_bboxes_map = {}  # reclip bboxes in cropped img rectangle.

    for processid, imgpath in enumerate(src_img_list):
        if len(src_img_list) > 100 and processid % (len(src_img_list) / 100) == 0:
            print("now at %s/%s .. " % (processid, len(src_img_list)))

        try:
            img = cv2.imread(imgpath)
        except:
            raise Exception("%s not exists" % (imgpath))
        xmlname = GET_BARENAME(imgpath) + '.xml'
        xml_path = os.path.join(src_xml_path, xmlname)
        # print("imgpath: %s, xmlname: %s, xml_path:%s"%(imgpath, xmlname, xml_path))
        pascal_voc_ann = PascalVocAnn(xml=xml_path)
        bboxes = pascal_voc_ann.get_boxes()
        h, w, c = pascal_voc_ann.get_size()
        boxes_list = []
        imgid = GET_BARENAME(imgpath)
        print("raw bboxes: ", bboxes)
        for (idx, box) in enumerate(bboxes):
            xmin, ymin, xmax, ymax = box[1:5]
            if imgid not in raw_imgid_bboxes_map:
                raw_imgid_bboxes_map[imgid] = set([])
            raw_imgid_bboxes_map[imgid].add(" ".join([str(b) for b in [xmin, ymin, xmax, ymax]]))
            if src_bboxes_type == 'square':
                tempbox = gen_square_from_rectangle(
                    src_bboxes=box[1:5],
                    src_refine_rectangle_size=src_refine_rectangle_size,
                    src_boundrary=[h, w]
                )
            elif src_bboxes_type == 'rectangle':
                tempbox = [xmin, ymin, xmax, ymax]
            else:
                raise Exception(
                    "%s is Not supported bbox's type, only support 'square' and 'rectangle'.." % (src_bboxes_type))
            boxes_list.append(tempbox)
        # print("%s with %s bboxes: %s" % (xml_path, len(boxes_list), boxes_list))

        for idx, box in enumerate(boxes_list):
            xmin, ymin, xmax, ymax = box
            wsize = xmax - xmin + 1
            hsize = ymax - ymin + 1
            xcenter = xmin + wsize / 2
            ycenter = ymin + hsize / 2
            # print("running.. gen expand width and height..")
            if src_bboxes_scale_shape == None:
                expand_width = src_img_output_size[0]
                expand_height = src_img_output_size[1]
            else:
                expand_width, expand_height = gen_expand_shape(src_img_out_shape_list=src_img_output_size,
                                                               src_bboxes_scale_shape=src_bboxes_scale_shape,
                                                               src_box_size_shape=[wsize, hsize])
            # print("running.. gen new tl br positions..")
            cxmin, cymin, cxmax, cymax = gen_xcyctoxy(src_center_list=[xcenter, ycenter],
                                                      src_expand_size=[expand_width, expand_height],
                                                      src_img_shape=[w, h])

            # if idx == 0:
            # print(
            #     "box%d with crop region [%s,%s,%s,%s], with expand [%s(w), %s(h)], center [%s, %s], with objects' [%s, %s, %s, %s]" % (
            #         idx, cxmin, cymin, cxmax, cymax, expand_width, expand_height, xcenter, ycenter, xmin, ymin, xmax,
            #         ymax))

            # debug
            # visionalize.visionalize_bboxes_list_on_img(
            #     src_img_file=imgpath,
            #     dst_img_file="/ssd/hnren/Data/dataset_pipe/newcrop_patches_int/test/" + GET_BARENAME(
            #         imgpath) + "_box%s.jpg" % (idx),
            #     src_bboxes_list=[[cxmin, cymin, cxmax, cymax]])

            # print("running.. crop, mask, or maybe resize..")
            dst_imgname = GET_BARENAME(imgpath) + "_crop%s" % (idx) + ".jpg"
            dst_img_path = os.path.join(dst_img_dir, dst_imgname)

            need_postprocess_boxes = get_related_bboxes_with_input_list(boxes_list, [cxmin, cymin, cxmax, cymax])
            tmp_img = img.copy()
            tmp_img = post_process_mask_bboxes(
                src_img=tmp_img,
                src_need_post_bboxes=need_postprocess_boxes
            )
            # for nbox in need_postprocess_boxes:
            #     if nbox[4] == 1:
            #         nbxmin, nbymin, nbxmax, nbymax = nbox[0:4]
            #         tmp_img[nbymin:nbymax, nbxmin:nbxmax] = 125
            img_crop = tmp_img[cymin:cymax, cxmin:cxmax]  # Tips: first start with y dim, then with x dim..

            if src_bboxes_scale_shape:
                img_crop = cv2.resize(img_crop, (src_img_output_size[0], src_img_output_size[1]),
                                      interpolation=cv2.INTER_CUBIC)
            img_crop_with = img_crop.shape[1]
            img_crop_height = img_crop.shape[0]
            assert img_crop_with <= src_img_output_size[0], "img_crop_with:%s with src_img_output_size[0]:%s" % (
                img_crop_with, src_img_output_size[0])
            assert img_crop_height <= src_img_output_size[1], "img_crop_height:%s with src_img_output_size[1]:%s" % (
                img_crop_height, src_img_output_size[1])
            assert img_crop_with == (cxmax - cxmin), "img_crop_with:%s with cxmin:%s, cxmax:%s" % (
                img_crop_with, cxmin, cxmax)
            assert img_crop_height == (cymax - cymin), "img_crop_height:%s with cymin:%s, cymax:%s" % (
                img_crop_height, cymin, cymax)
            cv2.imwrite(dst_img_path, img_crop)

            if imgid not in cropped_imgid_bboxes_map:
                cropped_imgid_bboxes_map[imgid] = set([])
            cropped_imgid_bboxes_map[imgid].add(" ".join([str(b) for b in [cxmin, cymin, cxmax, cymax]]))

            # print("running.. statisic related bboxes in this cropped img region..")
            rebased_need_box = []
            for boxid, boxlist in enumerate(need_postprocess_boxes):
                if boxlist[4] == 0:
                    [box_xmin, box_ymin, box_xmax, box_ymax] = boxlist[0:4]
                    reb_xmin = box_xmin - cxmin
                    reb_ymin = box_ymin - cymin
                    reb_xmax = box_xmax - cxmin
                    reb_ymax = box_ymax - cymin
                    reb_xmin = int(float(src_img_output_size[0]) / img_crop_with * reb_xmin)
                    reb_ymin = int(float(src_img_output_size[1]) / img_crop_height * reb_ymin)
                    reb_xmax = int(float(src_img_output_size[0]) / img_crop_with * reb_xmax)
                    reb_ymax = int(float(src_img_output_size[1]) / img_crop_height * reb_ymax)
                    rebased_need_box.append([reb_xmin, reb_ymin, reb_xmax, reb_ymax])

                    if imgid not in reclip_imgid_bboxes_map:
                        reclip_imgid_bboxes_map[imgid] = set([])
                    reclip_imgid_bboxes_map[imgid].add(
                        " ".join([str(b) for b in [box_xmin, box_ymin, box_xmax, box_ymax]]))

                    # if boxid == 0:
                    #     print(
                    #         "box%s with box [%s, %s, %s, %s] and crop img region [%s, %s, %s, %s] ,and rebased box [%s, %s, %s, %s]" %
                    #         (boxid, box_xmin, box_ymin, box_xmax, box_ymax, cxmin, cymin, cxmax, cymax, reb_xmin,
                    #          reb_ymin,
                    #          reb_xmax, reb_ymax)
                    #     )

            # print("running.. gen new xmls..")
            dst_xmlname = GET_BARENAME(imgpath) + "_crop%s" % (idx) + ".xml"
            dst_xml_path = os.path.join(dst_xml_dir, dst_xmlname)
            # print("dst_img_path: ", dst_img_path)
            newpascal_ann = PascalVocAnn(image_name=dst_img_path)
            newpascal_ann.set_filename(file_name=dst_img_path)
            newpascal_ann.set_size(size=[src_img_output_size[1], src_img_output_size[0], img.shape[2]])
            for reb_box in rebased_need_box:
                [reb_xmin, reb_ymin, reb_xmax, reb_ymax] = reb_box
                newpascal_ann.add_object(object_class="head", xmin=reb_xmin, ymin=reb_ymin, xmax=reb_xmax,
                                         ymax=reb_ymax)
            newpascal_ann.check_boxes()
            newpascal_ann.write_xml(dst_xml_path)

    out_raw_p = open(dst_raw_imgid_bboxes_map_file, 'w')

    for imgid, bboxes in raw_imgid_bboxes_map.items():
        str_bboxes = ','.join(bboxes)
        out_raw_p.write("%s,%s\n" % (imgid, str_bboxes))
    out_raw_p.close()

    out_crop_p = open(dst_cropped_imgid_bboxes_map_file, 'w')

    for imgid, bboxes in cropped_imgid_bboxes_map.items():
        str_bboxes = ','.join(bboxes)
        out_crop_p.write("%s,%s\n" % (imgid, str_bboxes))
    out_crop_p.close()

    out_reclip_p = open(dst_reclip_imgid_bboxes_map_file, 'w')

    for imgid, bboxes in reclip_imgid_bboxes_map.items():
        str_bboxes = ','.join(bboxes)
        out_reclip_p.write("%s,%s\n" % (imgid, str_bboxes))
    out_reclip_p.close()

    print("done..")


def gen_imgid_bboxes_map(src_xml_file):
    '''
    get bboxes of specific imgid.
    :param src_xml_file:
    :return:
        imgid, bboxe in str format line.
    '''
    if not os.path.exists(src_xml_file):
        raise Exception("%s not exists!" % (src_xml_file))

    imgid = GET_BARENAME(src_xml_file)
    pascal_voc_ann = PascalVocAnn(xml=src_xml_file)
    bboxes = pascal_voc_ann.get_boxes()
    h, w, c = pascal_voc_ann.get_size()
    boxes_list = []
    str_box_list = []
    for (idx, box) in enumerate(bboxes):
        xmin, ymin, xmax, ymax = box[1:5]
        str_box = "{}-{}-{}-{}".format(xmin, ymin, xmax, ymax)
        str_box_list.append(str_box)

    return imgid, "_".join(str_box_list)


def restore_from_str_box_line(src_str_box_line):
    '''
    restore from str box to bboxes.
    :param src_str_box_line:
    :return:
        list with element which is also list.
    '''
    words = src_str_box_line.strip().split('_')
    bboxes = []
    for wd in words:
        xmin, ymin, xmax, ymax = wd.split('-')
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes


def vis_bboxes_on_img(src_img_file, dst_img_file, src_bboxes_list):
    '''
    visionalize bboxes at img, then save it.
    :param src_img_file:
    :param dst_img_file:
    :param src_bboxes_list:
    :return:
    '''
    try:
        img = cv2.imread(src_img_file)
    except:
        raise Exception("imread fail with %s" % (src_img_file))
    val_rs = [0, 85, 170, 255]
    val_gs = [85, 170, 255, 0]
    val_bs = [170, 255, 0, 85]

    for idx, bbox in enumerate(src_bboxes_list):
        color_r = val_rs[idx % 4]
        color_g = val_gs[idx % 4]
        color_b = val_bs[idx % 4]
        color = (color_r, color_g, color_b)
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    cv2.imwrite(dst_img_file, img)


def gen_vis_with_jpg_xml(src_jpg_dir, src_xml_dir, dst_vis_dir):
    '''
    combine  `src jpg dir`  and  `src xml dir`  into  `dst vis dir`
    :param src_jpg_dir:
    :param src_xml_dir:
    :param dst_vis_dir:
    :return:
    '''
    imgspath = list_all_files(src_jpg_dir, exts=['jpg'])
    for img_file in imgspath:
        xmlname = GET_BARENAME(img_file) + ".xml"
        xml_file = os.path.join(src_xml_dir, xmlname)
        imgid, str_bbox_line = gen_imgid_bboxes_map(src_xml_file=xml_file)
        # print("xml_file: %s with str_bbox_line: %s"%(xmlname, str_bbox_line))
        bboxes = restore_from_str_box_line(src_str_box_line=str_bbox_line)
        visname = GET_BARENAME(img_file) + "_vis.jpg"
        vis_file = os.path.join(dst_vis_dir, visname)
        vis_bboxes_on_img(
            src_img_file=img_file,
            dst_img_file=vis_file,
            src_bboxes_list=bboxes
        )


def vis_imgid_associated_bboxes(src_img_dir, src_raw_map_file, src_crop_map_file, src_reclip_map_file):
    '''
    vis bboxes on img, according to the imgid-bboxes files.
      i) get bboxes list from imgid-bboxes file
     ii) call `visionalize_bboxes_list_on_img_dat` to gen embedded imgs.
    iii) save img.

    :param src_img_dir:
    :param src_raw_map_file:
    :param src_crop_map_file:
    :param src_reclip_map_file:
    :return:
    '''


def Test_gen_slice_patches_in_pascal_format():
    '''
    test gen slice patches in pascal format..
    test factory:
    /ssd/hnren/Data/dataset_pipe/newcrop_patches_int/source/
    /ssd/hnren/Data/dataset_pipe/newcrop_patches_int/source/JPEGImages/ch01011_20190322_ch01011_20190322084000.mp4.cut.mp4_003000.jpg
    /ssd/hnren/Data/dataset_pipe/newcrop_patches_int/source/Annotations/ch01011_20190322_ch01011_20190322084000.mp4.cut.mp4_003000.xml
    /ssd/hnren/Data/dataset_pipe/newcrop_patches_int/test/
    :return:
    '''
    imgs_dir = "/ssd/hnren/Data/dataset_pipe/newcropv2_patches_int/source/JPEGImages/"
    xmls_dir = "/ssd/hnren/Data/dataset_pipe/newcropv2_patches_int/source/Annotations/"
    newvocdir = "/ssd/hnren/Data/dataset_pipe/newcropv2_patches_int/test/"
    dst_img_dir = os.path.join(newvocdir, "JPEGImages")
    dst_xml_dir = os.path.join(newvocdir, "Annotations")
    dst_vis_dir = os.path.join(newvocdir, "JPEG_with_anno")

    dst_raw_map_file = newvocdir + os.sep + "raw_imgid_bboxes_map.csv"
    dst_crop_map_file = newvocdir + os.sep + "crop_imgid_bboxes_map.csv"
    dst_reclip_map_file = newvocdir + os.sep + "reclip_imgid_bboxes_map.csv"

    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
    if not os.path.exists(dst_xml_dir):
        os.makedirs(dst_xml_dir)
    if not os.path.exists(dst_vis_dir):
        os.makedirs(dst_vis_dir)

    img_output_size = 300

    gen_slice_patches_in_pascal_format(
        src_img_path=imgs_dir,
        src_xml_path=xmls_dir,
        dst_img_dir=dst_img_dir,
        dst_xml_dir=dst_xml_dir,
        src_img_output_size=[img_output_size, img_output_size],
        src_refine_rectangle_size=0.7,
        src_bboxes_type='rectangle',
        src_bboxes_scale_shape=None,
        dst_raw_imgid_bboxes_map_file=dst_raw_map_file,
        dst_cropped_imgid_bboxes_map_file=dst_crop_map_file,
        dst_reclip_imgid_bboxes_map_file=dst_reclip_map_file
    )
    gen_vis_with_jpg_xml(
        src_jpg_dir=dst_img_dir,
        src_xml_dir=dst_xml_dir,
        dst_vis_dir=dst_vis_dir
    )

    raw_map_file = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=dst_raw_map_file)
    print("raw_map_file: ", raw_map_file)

    crop_map_file = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=dst_crop_map_file)
    print("crop_map_file: ", crop_map_file)

    reclip_map_file = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=dst_reclip_map_file)
    print("reclip_map_file: ", reclip_map_file)

    for imgid, raw_bboxes_list in raw_map_file.items():
        imgpath = os.path.join(imgs_dir, imgid + ".jpg")
        imgpath_raw_save = os.path.join(dst_vis_dir, imgid + "_with_rawbox.jpg")
        imgpath_crop_save = os.path.join(dst_vis_dir, imgid + "_with_cropbox.jpg")
        imgpath_reclip_save = os.path.join(dst_vis_dir, imgid + "_with_reclip.jpg")
        imgpath_save = os.path.join(dst_vis_dir, imgid + "_with_all_bboxes.jpg")
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
            src_color_idx=0,
            src_color_width=1
        )
        allimg = visionalize_bboxes_list_on_img_dat(
            src_img_data=allimg,
            src_bboxes_list=raw_bboxes_list,
            src_color_idx=0,
            src_color_width=1
        )

        crop_bboxes_list = crop_map_file[imgid]
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

        reclip_bboxes_list = reclip_map_file[imgid]
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


def gen_patches_voc2voc_format(dataset_list, src_refine_rectangle_size, req_imgsize=300, img_output_size=300,
                               gen_gt_rect=False):
    anno_type = 1  # fully labelled
    for data_folder in dataset_list:
        ori_anns_folder = os.path.join(data_folder, "Annotations")
        ori_imgs_folder = os.path.join(data_folder, "JPEGImages")
        imgs = list_all_files(ori_imgs_folder, exts=["jpg"])
        xmls = list_all_files(ori_anns_folder, exts=["xml"])
        newvocdir = data_folder + "_patches_int"
        print("....crop_and_gen_pascal")
        all_imgs_num = len(imgs)
        for i, img_path in enumerate(imgs):
            if i % (all_imgs_num / 100) == 0:
                print("process %s/%s" % (i, all_imgs_num))

            img_base_name = os.path.basename(img_path)
            xml_base_name = os.path.splitext(img_base_name)[0] + ".xml"
            ori_xml = os.path.join(ori_anns_folder, xml_base_name)
            crop_and_gen_pascal(img_path=img_path,
                                ori_xml=ori_xml,
                                newvocdir=newvocdir,
                                img_output_size=img_output_size,
                                src_refine_rectangle_size=src_refine_rectangle_size,
                                gen_gt_rect=gen_gt_rect)


def gen_patches_dataset_in_pascal(
        dataset_list, src_refine_rectangle_size, req_imgsize=300, img_output_size=300,
        gen_gt_rect=False):
    anno_type = 1  # fully labelled
    for data_folder in dataset_list:
        ori_anns_folder = os.path.join(data_folder, "Annotations")
        ori_imgs_folder = os.path.join(data_folder, "JPEGImages")
        imgs = list_all_files(ori_imgs_folder, exts=["jpg"])
        xmls = list_all_files(ori_anns_folder, exts=["xml"])
        newvocdir = data_folder + "_patches_int"
        print("....crop_and_gen_pascal")
        all_imgs_num = len(imgs)
        # for i, img_path in enumerate(imgs):
        #     if i % (all_imgs_num / 100) == 0:
        #         print("process %s/%s" % (i, all_imgs_num))
        #
        #     img_base_name = os.path.basename(img_path)
        #     xml_base_name = os.path.splitext(img_base_name)[0] + ".xml"
        #     ori_xml = os.path.join(ori_anns_folder, xml_base_name)
        #     crop_and_gen_pascal(img_path=img_path,
        #                         ori_xml=ori_xml,
        #                         newvocdir=newvocdir,
        #                         img_output_size=img_output_size,
        #                         src_refine_rectangle_size=src_refine_rectangle_size,
        #                         gen_gt_rect=gen_gt_rect)

        gen_slice_patches_in_pascal_format(
            src_img_path=imgs,
            src_xml_path=xmls,
            dst_img_dir=os.path.join(newvocdir, "JPEGImages"),
            dst_xml_dir=os.path.join(newvocdir, "Annotations"),
            src_img_output_size=[img_output_size, img_output_size],
            src_refine_rectangle_size=0.7,
            src_bboxes_type='rectangle',
            src_bboxes_scale_shape=None
        )


def gen_positive_list_voc_format():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/fid_fullframe_lst.txt"
    dataset_list = ['/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_0',
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_1',
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_2',
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/HeadBoxDataFidChecked2']

    anno_type = 1  # fully labelled
    fout = open(out_file, "wt")
    for data_folder in dataset_list:
        ori_anns_folder = os.path.join(data_folder, "Annotations")
        ori_imgs_folder = os.path.join(data_folder, "JPEGImages")
        imgs = list_all_files(ori_imgs_folder, exts=["jpg"])
        for i, img_path in enumerate(imgs):
            img_base_name = os.path.basename(img_path)
            xml_base_name = os.path.splitext(img_base_name)[0] + ".xml"
            ori_xml = os.path.join(ori_anns_folder, xml_base_name)
            pascal_voc_ann = PascalVocAnn(xml=ori_xml)
            boxes = pascal_voc_ann.get_boxes()
            if len(boxes) == 0:
                continue

            vec = []
            for b in boxes:
                xmin, ymin, xmax, ymax = b[1:5]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                sz = int(max(w, h) * 0.62)
                x = int(xmin + (w - sz) * 0.5)
                y = int(ymin + h - sz)
                vec.extend([x, y, sz, sz])

            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, vec)) + "\n"
            fout.write(line)
    fout.close()


def gen_positive_list_format1():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/did_full_all_20190703.txt"
    dataset_list = [('/ssd/xulifeng/train_data/head_detection/did/did_20190609/part0_label.json',
                     '/ssd/xulifeng/train_data/head_detection/did/did_20190609/part0/images'),
                    ('/ssd/xulifeng/train_data/head_detection/did/did_20190609/part1_label.json',
                     '/ssd/xulifeng/train_data/head_detection/did/did_20190609/part1/images')]

    '''
    root_dir = "/ssd/xieqiang/Data/nav_tracking_benchmark"
    dataset2 = ['20190703_nav_tracking_head_airport_to_label_0',
                '20190703_nav_tracking_head_to_label_0',
                '20190703_nav_tracking_head_to_label_1',
                '20190703_nav_tracking_head_to_label_2', 
                '20190703_nav_tracking_head_to_label_3']
    for d in dataset2:
        folder = os.path.join(root_dir, d)
        dataset_list.append((os.path.join(folder,"output.json"), os.path.join(folder,"images")))
    '''

    anno_type = 1  # fully labelled
    fout = open(out_file, "wt")
    for (json_file, img_folder) in dataset_list:
        data = json.load(open(json_file, 'r'))
        for image_name in data.keys():
            boxes = data[image_name]
            if len(boxes) == 0:
                continue
            img_path = os.path.join(img_folder, image_name)
            vec = []
            for b in boxes:
                sz = int(max(b[2], b[3]) * 0.62)
                x = int(b[0] + (b[2] - sz) * 0.5)
                y = int(b[1] + b[3] - sz)
                vec.extend([x, y, sz, sz])

            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, vec)) + "\n"
            fout.write(line)
    fout.close()


def gen_positive_list_format2():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/head_crops_lst.txt"
    img_roots = [(0, '/ssd/xulifeng/misc/hd_trainset')]

    fout = open(out_file, "wt")
    for (anno_type, img_root) in img_roots:
        imgs = list_all_files(img_root, exts=['jpg', 'bmp', 'png'])

        for idx, img_path in enumerate(imgs):
            anno_file = os.path.splitext(img_path)[0] + ".head.txt"
            if not os.path.exists(anno_file):
                continue
            boxes = []
            for line in open(anno_file):
                b = map(int, line.strip().split(","))
                if len(b) < 4:
                    continue

                sz = int(max(b[2], b[3]) * 0.62)
                x = int(b[0] + (b[2] - sz) * 0.5)
                y = int(b[1] + b[3] - sz)
                boxes.extend([x, y, sz, sz])
            if len(boxes) < 4:
                continue
            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, boxes)) + "\n"
            fout.write(line)
    fout.close()


def gen_positive_list_from_landmarks():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/face_crops_lst.txt"
    img_roots = ['/ssd/xulifeng/misc/fd_trainset/part1']
    anno_type = 0

    imgs = []
    for img_root in img_roots:
        imgs.extend(list_all_files(img_root, exts=['jpg', 'bmp', 'png']))

    img_box_lst = []
    for img_path in imgs:
        anno_file = os.path.splitext(img_path)[0] + ".ldmk.txt"
        if not os.path.exists(anno_file):
            continue
        with open(anno_file) as f:
            ldmk = map(float, next(f).strip().split(","))
            b = get_bbox_from_ldmk(ldmk)[0:4]
            x = int(b[0] - (b[2] - 1) * 0.5)
            y = int(b[1] - (b[2] - 1) * 0.5)
            sz = int(b[2])
            img_box_lst.append((img_path, [x, y, sz, sz]))

    with open(out_file, "wt") as fout:
        for (img_path, boxes) in img_box_lst:
            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, boxes)) + "\n"
            fout.write(line)


def gen_all_positive_list():
    positives = ["/ssd/xulifeng/workspace/hd_densebox/train_data/face_crops_lst.txt",
                 "/ssd/xulifeng/workspace/hd_densebox/train_data/head_crops_lst.txt",
                 # "/ssd/xulifeng/workspace/hd_densebox/train_data/head_did_lst.txt"
                 ]
    workdir = "/ssd/xulifeng/workspace/hd_densebox/train_data"
    concat_files(positives, os.path.join(workdir, "positive_lst.txt"))


def gen_all_positive_list2():
    positives = ["/ssd/xulifeng/workspace/hd_densebox/train_data/did_fullframe_lst.txt",
                 "/ssd/xulifeng/workspace/hd_densebox/train_data/fid_fullframe_lst.txt",
                 ]
    workdir = "/ssd/xulifeng/workspace/hd_densebox/train_data"
    concat_files(positives, os.path.join(workdir, "all_fullframe_lst.txt"))


def get_max_bbox_size(dataset_list):
    res_max_size = 0
    for data_folder in dataset_list:
        ori_anns_folder = os.path.join(data_folder, "Annotations")
        xmls = list_all_files(ori_anns_folder, exts=["xml"])
        for i, xml_path in enumerate(xmls):
            pascal_voc_ann = PascalVocAnn(xml=xml_path)
            bboxes = pascal_voc_ann.get_boxes()
            vec = []
            for i, b in enumerate(bboxes):
                xmin, ymin, xmax, ymax = b[1:5]
                max_size = max(xmax - xmin, ymax - ymin)
                if res_max_size < max_size:
                    res_max_size = max_size

    return res_max_size


if __name__ == "__main__":
    # # dslist = ['HeadVocFormat/FID_DID_HEAD_CLEAN_0',
    # #           'HeadVocFormat/FID_DID_HEAD_CLEAN_1',
    # #           'HeadVocFormat/FID_DID_HEAD_CLEAN_2',
    # #           'HeadVocFormat/HeadBoxDataFidChecked2']
    # # dslist= ['HeadVocFormat/HeadBoxDataFidChecked2']
    # dslist = ['FID_DID_HEAD_CLEAN_0']
    # for dsitem in dslist:
    #     if not os.path.exists(dsitem):
    #         raise Exception("%s not exists" % (dsitem))
    #
    # max_sz = get_max_bbox_size(dataset_list=dslist)
    # print("max sz:", max_sz)
    # gen_patches_voc2voc_format(dataset_list=dslist, req_imgsize=int(max_sz * 1.5), img_output_size=300,
    #                            gen_gt_rect=False)
    #
    # pass

    # Test_gen_square_from_rectangle()
    Test_gen_slice_patches_in_pascal_format()

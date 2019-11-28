import os
import cv2


def checkpath(filepath):
    if not os.path.exists(filepath):
        raise Exception("%s not exists" % (filepath))


def visionalize_bboxes_list_on_img(src_img_file, dst_img_file, src_bboxes_list):
    checkpath(src_img_file)
    assert dst_img_file[-4:]=='.jpg'
    val_rs = [0, 85, 170, 255]
    val_gs = [85, 170, 255, 0]
    val_bs = [170, 255, 0, 85]

    imgraw = cv2.imread(src_img_file)
    img = imgraw.copy()
    for box_id, box in enumerate(src_bboxes_list):
        color_r = val_rs[box_id % 4]
        color_g = val_gs[box_id % 4]
        color_b = val_bs[box_id % 4]
        color = (color_r, color_g, color_b)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    cv2.imwrite(dst_img_file,img)

def visionalize_bboxes_list_on_img_dat(src_img_data, src_bboxes_list, src_color_idx, src_color_width):
    '''
    visionalize bboxes at given img_dat, then return the visionalized img_data(within bboxes).
    :param src_img_data:
    :param src_bboxes_list:
    :return:
    '''

    val_rs = [0, 85, 170, 255]
    val_gs = [85, 170, 255, 0]
    val_bs = [170, 255, 0, 85]

    img = src_img_data.copy()
    for box_id, box in enumerate(src_bboxes_list):
        color_r = val_rs[src_color_idx % 4]
        color_g = val_gs[src_color_idx % 4]
        color_b = val_bs[src_color_idx % 4]
        color = (color_r, color_g, color_b)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, src_color_width)

    return img
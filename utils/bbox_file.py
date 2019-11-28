import os


def get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file):
    '''
    gen imgid-bboxes
    bboxes may be two-stage list. aka , list[[],[],...]
    :param src_imgid_boxes_map_file:
    :return:
        imgid-bboxes map.
    '''

    if not os.path.exists(src_imgid_boxes_map_file):
        raise Exception("%s not exists" % (src_imgid_boxes_map_file))

    imgid_bboxes_map = {}
    with open(src_imgid_boxes_map_file, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(',')
            imgid = words[0]
            bboxes_list = []
            for strbox in words[1:]:
                box = strbox.split(' ')
                box = [int(b) for b in box]
                bboxes_list.append(box)
            imgid_bboxes_map[imgid] = bboxes_list

    return imgid_bboxes_map


def Test_get_imgid_bboxes_map_from_file():
    raw_imgid_map_filepath = "/ssd/hnren/Data/dataset_pipe/newcrop_patches_int/test/raw_imgid_bboxes_map.csv"
    res = get_imgid_bboxes_map_from_file(src_imgid_boxes_map_file=raw_imgid_map_filepath)
    print(res)


if __name__ == '__main__':
    Test_get_imgid_bboxes_map_from_file()

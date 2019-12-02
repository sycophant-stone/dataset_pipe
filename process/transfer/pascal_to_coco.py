# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import argparse



voc_clses = ['backgroud','head']


categories = []
for iind, cat in enumerate(voc_clses):
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind
    categories.append(cate)

def getimages(xmlname, id):
    sig_xml_box = []
    tree = ET.parse(xmlname)
    root = tree.getroot()
    images = {}
    for i in root:  # 遍历一级节点
        if i.tag == 'filename':
            file_name = i.text  # 0001.jpg
            # print('image name: ', file_name)
            images['file_name'] = os.path.basename(file_name)
        if i.tag == 'size':
            for j in i:
                if j.tag == 'width':
                    width = j.text
                    images['width'] = int(width)
                if j.tag == 'height':
                    height = j.text
                    images['height'] = int(height)
        if i.tag == 'object':
            for j in i:
                if j.tag == 'name':
                    cls_name = j.text
                cat_id = voc_clses.index(cls_name)# not need to  + 1
                if j.tag == 'bndbox':
                    bbox = []
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for r in j:
                        if r.tag == 'xmin':
                            xmin = eval(r.text)
                        if r.tag == 'ymin':
                            ymin = eval(r.text)
                        if r.tag == 'xmax':
                            xmax = eval(r.text)
                        if r.tag == 'ymax':
                            ymax = eval(r.text)
                    bbox.append(xmin)
                    bbox.append(ymin)
                    bbox.append(xmax - xmin)
                    bbox.append(ymax - ymin)
                    bbox.append(id)   # 保存当前box对应的image_id
                    bbox.append(cat_id)
                    # anno area
                    bbox.append((xmax - xmin) * (ymax - ymin) - 10.0)   # bbox的ares
                    # coco中的ares数值是 < w*h 的, 因为它其实是按segmentation的面积算的,所以我-10.0一下...
                    sig_xml_box.append(bbox)
                    # print('bbox', xmin, ymin, xmax - xmin, ymax - ymin, 'id', id, 'cls_id', cat_id)
    images['id'] = id
    # print ('sig_img_box', sig_xml_box)
    return images, sig_xml_box



def txt2list(txtfile):
    f = open(txtfile)
    l = []
    for line in f:
        l.append(line[:-1])
    return l

def run(src_pascal_xml_path, src_pascal_set_file_path, dst_coco_dir):
    '''

    :param src_pascal_xml_path:
    :param src_pascal_set_file_path:
    :param dst_coco_dir:
    :return:
    '''
    if not os.path.exists(dst_coco_dir):
        os.makedirs(dst_coco_dir)

    set_name = os.path.splitext(os.path.basename(src_pascal_set_file_path))[0]

    xml_names = txt2list(src_pascal_set_file_path)
    xmls = []
    bboxes = []
    ann_js = {}
    for ind, xml_name in enumerate(xml_names):
        xmls.append(os.path.join(src_pascal_xml_path, xml_name + '.xml'))
    json_name = '%s/instances_%s.json'%(dst_coco_dir,set_name)
    images = []
    for i_index, xml_file in enumerate(xmls):
        image, sig_xml_bbox = getimages(xml_file, i_index)
        images.append(image)
        bboxes.extend(sig_xml_bbox)
    ann_js['images'] = images
    ann_js['categories'] = categories
    annotations = []
    for box_ind, box in enumerate(bboxes):
        anno = {}
        anno['image_id'] =  box[-3]
        anno['category_id'] = box[-2]
        anno['bbox'] = box[:-3]
        anno['id'] = box_ind
        anno['area'] = box[-1]
        anno['iscrowd'] = 0
        annotations.append(anno)
    ann_js['annotations'] = annotations
    json.dump(ann_js, open(json_name, 'w'), indent=4)



"""
python xml2json.py --pascal_xmlspath=Annotations --pascal_settxtpath=ImageSets/Main/test.txt
"""

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--pascal_xmlspath', type=str, help='pascal xmls path', required=True)
    parse.add_argument('--pascal_settxtpath', type=str, help='images path', required=True)
    _args = parse.parse_args()
    input_xmls_path = _args.pascal_xmlspath
    input_set_path  = _args.pascal_settxtpath

    if not os.path.exists("annotations"):
        os.mkdir("annotations")

    set_name = os.path.splitext(os.path.basename(input_set_path))[0]
    # input_xmls_path = 'anns'
    #input_xmls_path = '/data2/chenjia/data/VOCdevkit/VOC2007/Annotations'
    # input_set_path = 'voc2007/test.txt'
    #input_set_path = '/data2/chenjia/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

    xml_names = txt2list(input_set_path)
    xmls = []
    bboxes = []
    ann_js = {}
    for ind, xml_name in enumerate(xml_names):
        xmls.append(os.path.join(input_xmls_path, xml_name + '.xml'))
    json_name = 'annotations/instances_%s.json'%(set_name)
    images = []
    for i_index, xml_file in enumerate(xmls):
        image, sig_xml_bbox = getimages(xml_file, i_index)
        images.append(image)
        bboxes.extend(sig_xml_bbox)
    ann_js['images'] = images
    ann_js['categories'] = categories
    annotations = []
    for box_ind, box in enumerate(bboxes):
        anno = {}
        anno['image_id'] =  box[-3]
        anno['category_id'] = box[-2]
        anno['bbox'] = box[:-3]
        anno['id'] = box_ind
        anno['area'] = box[-1]
        anno['iscrowd'] = 0
        annotations.append(anno)
    ann_js['annotations'] = annotations
    json.dump(ann_js, open(json_name, 'w'), indent=4)  # indent=4

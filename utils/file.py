import os
# list all image files
def list_all_files(dir_name, exts=['jpg', 'bmp', 'png', 'xml']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def GET_BARENAME(fullname):
    try:
        return os.path.splitext(os.path.basename(fullname))[0]
    except:
        raise Exception("%s os actions error "%(fullname))

def TRS_JPG_TO_XML_NAME(src_jpg_filename):
    '''
    transfer jpg img name to xml file name, also with ext..
    :param src_jpg_filename:
    :return:
    '''
    imgid = GET_BARENAME(src_jpg_filename)
    return imgid+".xml"

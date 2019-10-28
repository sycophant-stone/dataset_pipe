import os
import hashlib
import shutil

def file_get_contents(filename):
    with open(filename, 'rb') as f:
        return f.read(-1)


def file_put_contents(filename, data, binary_mode=False):
    mode = 'w'
    if binary_mode:
        mode += 'b'
    with open(filename, mode) as f:
        f.write(data)


def file_get_content_as_lines(filename):
    if filename.startswith('hdfs://'):
        filename = hdfs_get_file(filename[len('hdfs://'):], True)
    with open(filename, 'r') as f:
        return f.read().splitlines()


def file_num_of_dir(dir):
    if dir.startswith('hdfs://'):
        dir = hdfs_get_dir(dir, True)
    return sum([len(files) for r, d, files in os.walk(dir)])




def ensure_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def md5(filename):
    hash_md5 = hashlib.md5()
    if filename.startswith('hdfs://'):
        filename = hdfs_get_file(filename[len('hdfs://'):])
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def size(filename):
    if filename.startswith('hdfs://'):
        filename = hdfs_get_file(filename[len('hdfs://'):])
    return os.path.getsize(filename)

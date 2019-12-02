import os
import utils.shell as shell
def multifilecopy(src_file_dir, dst_file_dir):
    if not os.path.isdir(src_file_dir):
        raise Exception("%s is Not folder.. "%(src_file_dir))
    if not os.path.isdir(dst_file_dir):
        raise Exception("%s is Not folder.. " % (src_file_dir))
    src_files = os.listdir(src_file_dir)
    for src_file_name in src_files:
        src_file_path = os.path.join(src_file_dir, src_file_name)
        if not os.path.exists(src_file_path):
            raise Exception("%s not exists!"%(src_file_path))
        cmd = "cp %s %s"%(src_file_path, dst_file_dir)
        shell.run_system_command(cmd)

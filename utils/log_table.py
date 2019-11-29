from prettytable import PrettyTable


class LogTable(object):
    def __init__(self, src_table_head_list):
        self.table_head_list = src_table_head_list
        self.table_obj = PrettyTable(src_table_head_list)

    def add_line(self, line_list):
        '''
        add log line
        :param line_list:
        :return:
        '''
        self.table_obj.add_row(line_list)

    def show(self):
        '''
        show table log.
        :return:
        '''
        print(self.table_obj)

    def savefile(self, dst_logtable_file):
        '''
        save to file
        :param dst_logtable_file: 
        :return: 
        '''
        file = open(dst_logtable_file, 'w')
        file.write(str(self.table_obj))
        file.close()

        


def Test_gen_table():
    table = LogTable(["Item", "value"])
    table.add_line(["conf value", [0.26462996, 0.39294541, 0.49455839]])
    table.add_line(["iou  value", [0., 0., 0.77088083]])
    print(table)
    table.savefile(dst_logtable_file='logtable_expample.file')


if __name__ == '__main__':
    Test_gen_table()

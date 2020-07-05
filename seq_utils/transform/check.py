"""
    用于查看生成的txt.gz文件
"""

import os
import gzip

if __name__ == '__main__':
    file = input("Please input file:")
    while True:
        if os.path.exists(os.path.join('E:/Projects/RS/ESRT/indexed_data/seq_min_count5', file)):
            file = os.path.join('E:/Projects/RS/ESRT/indexed_data/seq_min_count5/', file)
            break
        elif os.path.exists(os.path.join('E:/Projects/RS/ESRT/indexed_data/seq_min_count5/seq_query_split', file)):
            file = os.path.join('E:/Projects/RS/ESRT/indexed_data/seq_min_count5/seq_query_split', file)
            break
        print("Wrong path!")
        file = input("Please input file:")

    with gzip.open(file, 'r') as fin:
        lines = []
        for line in fin:
            lines.append(str(line.strip()))
    try:
        while True:
            command = input("Please input the line index:(CTRL+D to quit):")
            print(lines[int(command)])
    except EOFError:
        pass

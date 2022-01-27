import os
def process_label(add_path,gt_file):
    gt_list = []
    with open(gt_file, 'r', encoding='UTF-8') as f:
        all = f.readlines()
        for each in all:
            each = add_path+each
            each = each.strip()+'\n'
            gt_list.append(each)
    print(gt_list)
    with open(gt_file, 'w', encoding='UTF-8') as f:
        f.writelines(gt_list)

def write_txt(gt_list,gt_file):
    with open(gt_file, 'w', encoding='UTF-8') as f:
        f.writelines(gt_list)

def strip_label(gt_file):
    gt_list = []
    with open(gt_file, 'r', encoding='UTF-8') as f:
        all = f.readlines()
        for each in all:
            each = each.strip()
            gt_list.append(each+'\n')
    with open(gt_file, 'w', encoding='UTF-8') as f:
        f.writelines(gt_list)

def dict_label(gt_file):
    val_label = '../train_data/ppdataset/train/labelval.txt'
    dict_path = '../ppocr/utils/ppocr_keys_v2.txt'
    # val_label  = 'labelval.txt'
    gt_list = []
    max_len = 0
    dict_char = {}
    set_char = set()
    set_val = set()
    with open(gt_file, 'r', encoding='UTF-8') as f:
        all= f.readlines()
        for each in all:
            origin_label = each
            each = each.strip().split('\t')
            text = each[1]
            for i in text:
                set_char.add(i+'\n')
                value = dict_char.setdefault(i, 0)
                if (value < 60 and value % 20 == 0 ) or value % 500 == 0:
                    set_val.add(origin_label)
                value +=1
                dict_char.update({i:value})
            max_len = len(text) if max_len<len(text) else max_len
            gt_list.append(text)
    print("max_len:{}".format(max_len))
    # print("char_dict:{}".format(dict_char))
    # print("char_len:{}".format(len(set_char)))
    print("val_line:{}".format(set_val))
    # write_txt(set_char,dict_path)
    write_txt(set_val,val_label)

if __name__ == '__main__':
    gt_file = '../datasets/NewVersion/val_data/CUTE.txt'
    add_path = 'CUTE/'
    baseline_file = '../output/rec/ztl_baseline.txt'
    # val_file = 'labelval.txt'
    process_label(add_path,gt_file)
    # dict_label(gt_file)
    # strip_label(gt_file)
    # strip_label(val_file)
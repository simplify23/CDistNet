import sys
import os
import editdistance
import pdb

### for icdar2003, svt, iiit5k, they all have lexicon_50
def ic03_lex(lexdir, lex_type, gt_file, submit_file, name_file):
    gt = open(gt_file, 'r')
    gt_lines = gt.readlines()
    gt.close()

    sub = open(submit_file, 'r')
    sub_lines = sub.readlines()
    sub.close()

    imgname = open(name_file, 'r')
    img_lines = imgname.readlines()
    imgname.close()

    # for lexicon full
    if lex_type== 'full':
        sub_file = submit_file[:-4] + '_full.txt'
        sub_fout = open(sub_file, 'w')
        for i in range(len(gt_lines)):#for each gt
            suf, gt, _ = (gt_lines[i].strip()).split('"')
            sub = (sub_lines[i].strip()).split('"')[1]

            lex_file = open(os.path.join(lexdir, 'lexicon_Full.txt'), 'r')
            lex = lex_file.readlines()
            lex_file.close()

            min_dis = 10000
            min_word = sub
            for word in lex:
                word = word.strip()
                word = word.lower()
                dis = editdistance.eval(sub, word)
                if dis < min_dis:
                    min_word = word
                    min_dis = dis
            sub_fout.write(suf + '"' + str(min_word) + '"\n')
        sub_fout.close()
    # for lexicon 50
    else:
        sub_file = submit_file[:-4] + '_50.txt'
        sub_fout = open(sub_file, 'w')
        for i in range(len(gt_lines)):#for each gt
            base_name = img_lines[i].strip().split('.')[0]
            suf, gt, _ = (gt_lines[i].strip()).split('"')
            sub = (sub_lines[i].strip()).split('"')[1]

            lex_file = open(os.path.join(lexdir, 'lexicon_50', 'lexicon_' + base_name + '_' + gt + '.txt'), 'r')
            lex = lex_file.readlines()
            lex_file.close()

            min_dis = 10000
            min_word = sub
            for word in lex:
                word = word.strip()
                word = word.lower()
                dis = editdistance.eval(sub, word)
                if dis < min_dis:
                    min_word = word
                    min_dis = dis
            sub_fout.write(suf + '"' + str(min_word) + '"\n')
        sub_fout.close()

    return sub_file

### for svt
def svt_lex(lexdir, lex_type, gt_file, submit_file, name_file):
    gt = open(gt_file, 'r')
    gt_lines = gt.readlines()
    gt.close()

    sub = open(submit_file, 'r')
    sub_lines = sub.readlines()
    sub.close()

    imgname = open(name_file, 'r')
    img_lines = imgname.readlines()
    imgname.close()

    # for lexicon 50
 
    sub_file = submit_file[:-4] + '_50.txt'
    sub_fout = open(sub_file, 'w')
    for i in range(len(gt_lines)):#for each gt
        base_name = img_lines[i].strip().split('.')[0]
        suf, gt, _ = (gt_lines[i].strip()).split('"')
        sub = (sub_lines[i].strip()).split('"')[1]

        lex_file = open(os.path.join(lexdir, 'lexicon_50', 'lexicon_' + base_name + '_' + gt + '.txt'), 'r')
        lex = lex_file.readlines()
        lex_file.close()

        min_dis = 10000
        min_word = sub
        for word in lex:
            word = word.strip()
            word = word.lower()
            dis = editdistance.eval(sub, word)
            if dis < min_dis:
                min_word = word
                min_dis = dis
        sub_fout.write(suf + '"' + str(min_word) + '"\n')
    sub_fout.close()

    return sub_file


### for svt-p
def svt_p_lex(lexdir, lex_type, gt_file, submit_file, name_file):
    gt = open(gt_file, 'r')
    gt_lines = gt.readlines()
    gt.close()

    sub = open(submit_file, 'r')
    sub_lines = sub.readlines()
    sub.close()

    imgname = open(name_file, 'r')
    img_lines = imgname.readlines()
    imgname.close()

    # for lexicon 50
 
    sub_file = submit_file[:-4] + '_50.txt'
    sub_fout = open(sub_file, 'w')
    for i in range(len(gt_lines)):#for each gt
        base_name = img_lines[i].strip().split('.')[0]
        suf, gt, _ = (gt_lines[i].strip()).split('"')
        sub = (sub_lines[i].strip()).split('"')[1]

        lex_file = open(os.path.join(lexdir, 'lexicon_50', 'lexicon_' + base_name + '_' + gt + '.txt'), 'r')
        lex = lex_file.readlines()
        lex_file.close()

        min_dis = 10000
        min_word = sub
        for word in lex:
            word = word.strip()
            word = word.lower()
            dis = editdistance.eval(sub, word)
            if dis < min_dis:
                min_word = word
                min_dis = dis
        sub_fout.write(suf + '"' + str(min_word) + '"\n')
    sub_fout.close()

    return sub_file

### for iiit5k
def iiit5k_lex(lexdir, lex_type, gt_file, submit_file, name_file):
    gt = open(gt_file, 'r')
    gt_lines = gt.readlines()
    gt.close()

    sub = open(submit_file, 'r')
    sub_lines = sub.readlines()
    sub.close()

    imgname = open(name_file, 'r')
    img_lines = imgname.readlines()
    imgname.close()

    # for lexicon full
    if lex_type== '1k':
        sub_file = submit_file[:-4] + '_1k.txt'
        sub_fout = open(sub_file, 'w')
        for i in range(len(gt_lines)):#for each gt
            base_name = img_lines[i].strip().split('.')[0]
            suf, gt, _ = (gt_lines[i].strip()).split('"')
            sub = (sub_lines[i].strip()).split('"')[1]

            lex_file = open(os.path.join(lexdir, 'lexicon_1k', 'lexicon_' + base_name + '_' + gt + '.txt'), 'r')
            lex = lex_file.readlines()
            lex_file.close()

            min_dis = 10000
            min_word = sub
            for word in lex:
                word = word.strip()
                word = word.lower()
                dis = editdistance.eval(sub, word)
                if dis < min_dis:
                    min_word = word
                    min_dis = dis
            sub_fout.write(suf + '"' + str(min_word) + '"\n')
        sub_fout.close()
    # for lexicon 50
    else:
        sub_file = submit_file[:-4] + '_50.txt'
        sub_fout = open(sub_file, 'w')
        for i in range(len(gt_lines)):#for each gt
            base_name = img_lines[i].strip().split('.')[0]
            suf, gt, _ = (gt_lines[i].strip()).split('"')
            sub = (sub_lines[i].strip()).split('"')[1]

            lex_file = open(os.path.join(lexdir, 'lexicon_50', 'lexicon_' + base_name + '_' + gt + '.txt'), 'r')
            lex = lex_file.readlines()
            lex_file.close()

            min_dis = 10000
            min_word = sub
            for word in lex:
                word = word.strip()
                word = word.lower()
                dis = editdistance.eval(sub, word)
                if dis < min_dis:
                    min_word = word
                    min_dis = dis
            sub_fout.write(suf + '"' + str(min_word) + '"\n')
        sub_fout.close()

    return sub_file
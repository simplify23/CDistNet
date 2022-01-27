import sys
import os
import editdistance

datatype = sys.argv[1]
lexdir = '../Dataset/TextRecog'
lexdir = os.path.join(lexdir, datatype)


### for icdar2003, svt, iiit5k, they all have lexicon_50
gt = open(sys.argv[2], 'r')
gt_lines = gt.readlines()
gt.close()

sub = open(sys.argv[3], 'r')
sub_lines = sub.readlines()
sub.close()

imgname = open(sys.argv[4], 'r')
img_lines = imgname.readlines()
imgname.close()

sub_50 = open(((sys.argv[3])[:-4] + '_50.txt'), 'w')

for i in range(len(gt_lines)):#for each gt
    img = img_lines[i].strip()
    gt = (gt_lines[i].strip()).split('\"')[1]
    sub = (sub_lines[i].strip()).split('\"')[1]

    lex_file = open(os.path.join(lexdir, 'lexicon_50', 'lexicon_' + img + '_' + gt + '.txt'), 'r')
    lex = lex_file.readlines()
    lex_file.close()

    min_dis = 10000
    min_word = sub
    for word in lex:
        word = word.strip()
        word = word.lower()
        dis = editdistance.eval(sub, word)
        if min_dis > dis:
            min_word = word
            min_dis = dis
    sub_50.write('word_' + str(i+1) + '.png, \"' + str(min_word) + '\"\n')
sub_50.close()



### for icdar2003, it has lexicon_Full
if datatype == 'icdar2003':
    gt = open(sys.argv[2], 'r')
    gt_lines = gt.readlines()
    gt.close()

    sub = open(sys.argv[3], 'r')
    sub_lines = sub.readlines()
    sub.close()

    imgname = open(sys.argv[4], 'r')
    img_lines = imgname.readlines()
    imgname.close()

    sub_Full = open(((sys.argv[3])[:-4] + '_Full.txt'), 'w')
    for i in range(len(gt_lines)):#for each gt
        img = img_lines[i].strip()
        gt = (gt_lines[i].strip()).split('\"')[1]
        sub = (sub_lines[i].strip()).split('\"')[1]

        lex_file = open(os.path.join(lexdir, 'lexicon_Full.txt'), 'r')
        lex = lex_file.readlines()
        lex_file.close()

        min_dis = 10000
        min_word = sub
        for word in lex:
            word = word.strip()
            word = word.lower()
            dis = editdistance.eval(sub, word)
            if min_dis > dis:
                min_word = word
                min_dis = dis
        sub_Full.write('word_' + str(i+1) + '.png, \"' + str(min_word) + '\"\n')
    sub_Full.close()

elif datatype == 'iiit5k':## for iiit5k, it has lexicon_1k
    gt = open(sys.argv[2], 'r')
    gt_lines = gt.readlines()
    gt.close()

    sub = open(sys.argv[3], 'r')
    sub_lines = sub.readlines()
    sub.close()

    imgname = open(sys.argv[4], 'r')
    img_lines = imgname.readlines()
    imgname.close()

    sub_1k = open(((sys.argv[3])[:-4] + '_1k.txt'), 'w')
    for i in range(len(gt_lines)):#for each gt
        img = img_lines[i].strip()
        gt = (gt_lines[i].strip()).split('\"')[1]
        sub = (sub_lines[i].strip()).split('\"')[1]

        lex_file = open(os.path.join(lexdir, 'lexicon_1k', 'lexicon_' + img + '_' + gt + '.txt'), 'r')
        lex = lex_file.readlines()
        lex_file.close()

        min_dis = 10000
        min_word = sub
        for word in lex:
            word = word.strip()
            dis = editdistance.eval(sub, word)
            if min_dis > dis:
                min_word = word
                min_dis = dis
        sub_1k.write('word_' + str(i+1) + '.png, \"' + str(min_word) + '\"\n')
    sub_1k.close()

dst_vocab = '/data6/zhengtianlun/temp/dict_99.txt'  # 98 + 空格
dst_vocab_size = 99
rgb2gray = True
keep_aspect_ratio = False
width = 96 #100
height = 32 #32
max_width = 180
is_lower = False  # True：训练的时候转成小写
cnn_num = 2
leakyRelu = False
hidden_units = 512
ff_units = 1024
scale_embedding = True
attention_dropout_rate = 0.0
residual_dropout_rate = 0.1
num_encoder_blocks = 4
num_decoder_blocks = 4
num_heads = 8
beam_size = 10
n_best = 1
num_fiducial = 20           #number of fiducial points of TPS-STN
use_squ = True              #if fasle: use diag for tgt mask(not ready)
train_method = 'origin'     #dist:  use distributed train method origin

# method choice
tps_block = None            # TPS None
feature_block = 'origin'    # None (not use cnn) origin Resnet
patch_block = 'wh'         # None wh_2_4_8 wh w+h+wh+avg vit w+h w+h+wh wh_fusion
custom_encoder = 'trans_blstm'    # None swin-trans pvt text2img-msa(not ready)
custom_decoder = None
transformer = 'transformer' # transformer patch4_trans


train = dict(
    grads_clip=5,
    optimizer='adam_decay',  # not used
    learning_rate_warmup_steps=10000,
    label_smoothing=True,  # fixed in code
    shared_embedding=False,  # not used
    device='cuda',
    # image_dir='/home/zhengsheng/datasets/TextRecog/mnt/ramdisk/max/90kDICT32px',
    # gt_file='/home/zhengsheng/datasets/TextRecog/mnt/ramdisk/max/90kDICT32px/annotation_train_two_Synth_shuf_clean.txt',
    image_dir='/data6/zhengtianlun/temp/icdar2015_test/images',
    gt_file='/data6/zhengtianlun/temp/icdar2015_test/gt.txt',
    # hdf5='datasets/train.hdf5',  # train_two.hdf5 train_keep_aspect_ratio.hdf5  train_two_keep_aspect_ratio.hdf5
    num_worker=16,
    model_dir='models/baseline_debug',  # 模型保存的目录
    num_epochs=4,
    # gpu_device_ids=[0],
    batch_size=250,  # 4gpu 1800
    model=None,  # 加载的模型地址, None不加载  e.g. '/home/zhengsheng/github/NRTR/models/model_epoch_14.pth',
    current_epoch=15,  # 从第几个epoch开始训练,根据加载的模型设置  e.g. 15
    save_iter=2000,
    display_iter=100,
    tfboard_iter=100,
    eval_iter=10,
)


val = dict(
    model='models/baseline_two_32*100_1d_2cnn-test/model_epoch_1.pth',  # 加载的模型, 训练的时候用不到
    device='cuda',
    image_dir='/data6/zhengtianlun/temp/icdar2015_test/images',
    gt_file='/data6/zhengtianlun/temp/icdar2015_test/gt.txt',
    # hdf5='datasets/val.hdf5',
    batch_size=1800,  # 4gpu 1800
    num_worker=16,
)


test = dict(
    device='cuda',
    rotate=False,  # 测试时旋转90度
    eval_all=False,  # 测试全部，包括两个epoch之间保存的模型。例如model_epoch_9_iter_4080.pth
    s_epoch=15,  # 从第s_epoch开始测试，当s_epoch = -1: 不测试
    e_epoch=15,  # 到第e_epoch结束测试
    avg_s=-1,  # 从第avg_s到avg_e进行模型平均，当avg_s = -1: 不平均
    avg_e=9,
    avg_all=False,  # 如果True，模型平均的时候包括两个epoch之间保存的模型。例如model_epoch_9_iter_4080.pth
    test_list=[
        'ICDAR2003_860',
        'ICDAR2003_867',
        'ICDAR2013_857',
        'ICDAR2013_1015',
        'ICDAR2015_1811',
        'ICDAR2015_2077',
        'IIIT5K',
        'SVT',
        'SVT-P',
        'CUTE80'
    ],
    image_dir='datasets/NewVersion',
    batch_size=8,
    num_worker=1,
    model_dir='models/baseline_20_epoch_trans_blstm_4*4',  # 测试加载的模型目录
    script_path='/data6/zhengtianlun/temp/Evaluation_TextRecog/script.py',
    python_path='/data1/zs/anaconda3/envs/py2/bin/python'
)

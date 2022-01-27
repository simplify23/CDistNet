dst_vocab = 'cdistnet/utils/dict_36.txt'  # 98 + 空格
dst_vocab_size = 40
rgb2gray =False
keep_aspect_ratio = False
width = 128 #100
height = 32 #32
max_width = 180
is_lower = True  # True：训练的时候转成小写
cnn_num = 2
leakyRelu = False
hidden_units = 512
ff_units = 1024      #ff的隐藏层数512 1024
scale_embedding = True
attention_dropout_rate = 0.0
residual_dropout_rate = 0.1
num_encoder_blocks = 3
num_decoder_blocks = 3
num_heads = 8
beam_size = 10
n_best = 1
data_aug = True
num_fiducial = 20           #number of fiducial points of TPS-STN
train_method = 'origin'     #dist:  use distributed train method origin
optim = 'origin'

# method choice
tps_block = 'TPS'      # TPS None
feature_block = 'Resnet45'    # Resnet45 Resnet31 MTB

train = dict(
    grads_clip=5,
    optimizer='adam_decay',  # not used
    learning_rate_warmup_steps=10000,
    label_smoothing=True,  # fixed in code
    shared_embedding=False,  # not used
    device='cuda',
    gt_file=['../dataset/MJ/MJ_train/',
            '../dataset/MJ/MJ_test/',
            '../dataset/MJ/MJ_valid/',
            '../dataset/ST'],
    num_worker=16,
    model_dir ='model/test',
    # model_dir='models/reconstruct_CDistNet_3_10',  # 模型保存的目录
    num_epochs=10,
    # gpu_device_ids=[1,2,3,4,5,6,7],
    batch_size=7,  # 4gpu 1800
    model=None,
    # model ='models/new_baseline_sem_pos_pos_vis_3_32*128_tps_resnet45_epoch_6/model_epoch_5.pth',
    # model='/data6/zhengtianlun/NRTR/models/new_baseline_transformer_4_32*128_tps_resnet45_epoch_5/model_epoch_3.pth',  # 加载的模型地址, None不加载  e.g. '/home/zhengsheng/github/NRTR/models/model_epoch_14.pth',
    # current_epoch=6,  # 从第几个epoch开始训练,根据加载的模型设置  e.g. 15
    save_iter=10000,
    display_iter=100,
    tfboard_iter=100,
    eval_iter=3000,
)


val = dict(
    model='models/baseline_two_32*100_1d_2cnn-test/model_epoch_1.pth',  # 加载的模型, 训练的时候用不到
    device='cuda',
    # is_val_gt=True,
    image_dir='datasets/NewVersion/val_data',
    gt_file= [
               '../dataset/eval/IC13_857',
               '../dataset/eval/SVT',
                '../dataset/eval/IIIT5k_3000',
               '../dataset/eval/IC15_1811',
                '../dataset/eval/SVTP',
               '../dataset/eval/CUTE80'],
    # gt_file=['datasets/NewVersion/val_data/val_data.txt'],
    # gt_file='../dataset/MJ/MJ_valid/',
    batch_size=800,  # 4gpu 1800
    num_worker=16,
)


test = dict(
    test_one=False,
    device='cuda',
    rotate=False,  # 测试时旋转90度
    best_acc_test=True,  # 测试评估模型中最好的结果
    eval_all=False,  # 测试全部，包括两个epoch之间保存的模型。例如model_epoch_9_iter_4080.pth
    s_epoch=7,  # 从第s_epoch开始测试，当s_epoch = -1: 不测试
    e_epoch=10,  # 到第e_epoch结束测试
    avg_s=-1,  # 从第avg_s到avg_e进行模型平均，当avg_s = -1: 不平均
    avg_e=9,
    avg_all=False,  # 如果True，模型平均的时候包括两个epoch之间保存的模型。例如model_epoch_9_iter_4080.pth
    is_test_gt=False,
    image_dir= None,     #if is_test_gt == False,needn't use image_dir
    test_list=[
               '../dataset/eval/IC13_857',
               '../dataset/eval/SVT',
                '../dataset/eval/IIIT5k_3000',
               '../dataset/eval/IC15_1811',
                '../dataset/eval/SVTP',
               '../dataset/eval/CUTE80'
    ],
    batch_size=128,
    num_worker=8,
    model_dir='models/reconstruct_CDistNetv3_3_10',  # 测试加载的模型目录
    script_path='utils/Evaluation_TextRecog/script.py',
    python_path='/data1/zs/anaconda3/envs/py2/bin/python'
)

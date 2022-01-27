import cv2
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import numpy as np

def gen_img(str1):
    '''
    text2img
    :param str1: string:text
    :return: ndarray:array
    '''
    # str1 = 'fdsakjhfdskjfh'
    num=len(str1)+1
    print(num)
    img = Image.new('RGB', (15*num,32), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    fontpath = "simsun.ttc"
    font = ImageFont.truetype(fontpath, 32)

    #绘制文字信息
    draw.text((0, 0),  str1, font = font, fill = (255,255,255))
    img = np.array(img)
    img=cv2.resize(img,(96,32))
    # plt.imshow(img)
    return img
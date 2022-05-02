import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst

def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)








f_c=open('carry_ids.txt','r')
f_s=open('skate_ids.txt','r')

carry_list=[]
skate_list=[]
line=f_c.readline()
while True:
    line=f_c.readline()
    if not line:
        break
    raw=line.rstrip("\n")
    carry_list.append(int(raw))
f_c.close()
line=f_s.readline()
while True:
    line=f_s.readline()
    if not line:
        break
    raw=line.rstrip("\n")
    skate_list.append(int(raw))
f_s.close()
random_c=random.sample(carry_list,16)
random_s=random.sample(skate_list,16)

for i in range(len(random_c)):
    globals()['im'+str(i)]=Image.open('results_imgs/'+str(random_c[i])+'gp.jpg').resize((480,480))
    globals()['img'+str(i)]=Image.open('results_imgs/'+str(random_s[i])+'gp.jpg').resize((480,480))


get_concat_tile_resize([[im0, im1, im2, im3],
                        [im4, im5, im6, im7],
                        [im8, im9, im10, im11],
                        [im12, im13, im14, im15]]).save('carry.jpg')

get_concat_tile_resize([[img0, img1, img2, img3],
                        [img4, img5, img6, img7],
                        [img8, img9, img10, img11],
                        [img12, img13, img14, img15]]).save('skate.jpg')


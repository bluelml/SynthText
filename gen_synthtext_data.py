"""
Gen data from CRNN training.
"""
from __future__ import division
import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import h5py
from common import *

from PIL import Image
from os import path
from random import shuffle



def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    # plt.figure(1)
    plt.figure(figsize=(50, 100))
    plt.imshow(text_im)
    plt.hold(True)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    for i in range(len(charBB_list)):
        bbs = charBB_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)



def get_bbox(wordBB):
    bboxes = []
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb, bb[:, 0]]
        box_x = []
        box_y = []
        for j in range(4):
            box_x.append(bb[0,j])
            box_y.append(bb[1,j])

        x_min = min(box_x)
        x_max = max(box_x)
        y_min = min(box_y)
        y_max = max(box_y)

        # left, top, right, bottom
        box = (x_min, y_min, x_max, y_max)
        bboxes.append(box)

    return bboxes


def array_to_list(txt):
    b = []
    for t in txt:
        d = t.decode("utf-8")
        e = d.split('\n')
        b.extend(e)

    res = []
    for s in b:
        s = s.lstrip(' ')
        s = s.rstrip(' ')
        res.append(s)

    res = ' '.join(res)
    res = res.split(' ')
    return res



def gen_datasets(datasets, db, out_dir):
    assert osp.exists(out_dir)
    out_file = osp.join(out_dir, 'sample.txt')
    with open(out_file, 'w') as f:
        for k in datasets:
            rgb = db['data'][k][...]
            wordBB = db['data'][k].attrs['wordBB']
            txt = db['data'][k].attrs['txt']
            print(type(txt))
            print(txt)
            # text = ' '.join([e for e in txt])
            # text = text.replace('\n', ' ')
            test = array_to_list(txt)
            print(test)

            bboxes = get_bbox(wordBB)
            image = Image.fromarray(rgb)
            for i, (t, b) in enumerate(zip(test, bboxes)):
                fn = k+'_'+str(i)+'.jpg'
                save_file = osp.join(out_dir, fn)
                img = image.crop(b)
                img.save(save_file)
                content = fn+' '+t
                print("content:", content)
                f.write(content)
                f.write('\n')


def main(db_fname, output_dir):
    if not path.exists(output_dir):
        os.system("mkdir -p {}".format(output_dir))

    train_d = path.join(output_dir, 'Train')
    test_d = path.join(output_dir, 'Test')
    os.system("mkdir -p {}".format(train_d))
    os.system("mkdir -p {}".format(test_d))

    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))

    shuffle(dsets)
    total = len(dsets)
    split = int(total*0.1)
    test_sets = dsets[:split]
    train_sets = dsets[split:]

    gen_datasets(train_sets, db, train_d)
    gen_datasets(test_sets, db, test_d)

    db.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Gen data')
    parser.add_argument('-f', '--h5_file', type=str, required=True,
                        help='Path to h5 file')
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                        help='Directory to store data')

    args = parser.parse_args()
    main(args.h5_file, args.data_dir)


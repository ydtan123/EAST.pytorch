#!/usr/bin/python3
## Libraries
import argparse
import math
import os
import pathlib
import sys

import numpy as np
import cv2


def getLocation(txtfile, imgw, imgh):
    #0 0.056711 0.629032 0.030246 0.204301
    symbols = []
    with open(str(txtfile)) as f:
        for line in f:
            data = [d for d in line.split(' ')]
            if (len(data) < 5):
                print("Incorrect data {0} in {1}".format(line, txtfile))
                continue
            dval = int(data[0])
            if (dval < 0 or dval > 9):
                print("Invalid number {0} in {1}".format(data[0], txtfile))
                continue
            cx = float(data[1]) * imgw
            cy = float(data[2]) * imgh
            w = float(data[3]) * imgw
            h = float(data[4]) * imgh
            symbols.append((data[0], cx, cy, w/2, h/2))
    symbols_sorted = sorted(symbols, key=lambda x: x[1])
    words = []
    for s in symbols_sorted:
        ty = s[2] - s[4]
        tx = s[1] - s[3]
        by = s[2] + s[4] 
        bx = s[1] + s[3]
        found = False
        for w in words:
            if (math.fabs(ty - w[2]) < s[4] and math.fabs(tx - w[3]) < s[3] * 4):
                w[0] += s[0]
                if (ty < w[2]):
                    w[2] = int(ty)
                w[3] = int(bx)
                if (by > w[4]):
                    w[4] = int(by)
                found = True
                break
        if (not found):
            words.append([s[0], int(tx), int(ty), int(bx), int(by)])
    return words

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
parser.add_argument("-d", "--debug", type=int, help="Debug mode", default=0)
parser.add_argument("-m", "--max", type=int, help="Maximum images to process", default=100000)
parser.add_argument("-p", "--prepare", type=str, help="Prepare training and testing sets", default='')
parser.add_argument("-s", "--save", action='store_true', help="Save text area")
args = vars(parser.parse_args())

if (args["prepare"] != ''):
    d = args["prepare"].split(',')
    train_size, test_size = int(d[0]), int(d[1])
    trn = 0
    tst = 0
    for f in pathlib.Path(args["image"]).glob("*.jpg"):
        if (trn >= train_size and tst > test_size):
            break
        img = cv2.imread(str(f))
        new_img = np.zeros((784, 1280, 3), dtype=np.uint8)
        new_img[0:img.shape[0], 0:img.shape[1]] = img
        if (args["debug"] > 0):
            cv2.imshow("img", new_img)
            cv2.waitKey(0)
        words = getLocation(f.with_suffix(".txt"), img.shape[1], img.shape[0])

        if (trn < train_size):
            cv2.imwrite("./dataset/train/img/img_{0}.jpg".format(trn), new_img)
            with open("./dataset/train/gt/img_{0}.txt".format(trn), 'w') as fgt:
                for w in words:
                    print(words)
                    fgt.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(w[1], w[2], w[3], w[2], w[3], w[4], w[1], w[4], w[0]))
            print("Train: {0} -> {1}".format(f, trn))
            trn += 1
        elif (tst < test_size):
            cv2.imwrite("./dataset/test/img/img_{0}.jpg".format(tst), new_img)
            with open("./dataset/test/gt/img_{0}.txt".format(tst), 'w') as fgt:
                for w in words:
                    print(words)
                    fgt.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(w[1], w[2], w[3], w[2], w[3], w[4], w[1], w[4], w[0]))
            print("Test: {0} -> {1}".format(f, tst))
            tst += 1
    exit(0)

processed = 0
for f in pathlib.Path(args["image"]).glob("*.jpg"):
    if (args["debug"] > 0):
        print("------------Image {0}---------------".format(processed))
        print(f)
    img = cv2.imread(str(f))
    words = getLocation(f.with_suffix(".txt"), img.shape[1], img.shape[0])
    if (args["debug"] > 0):
        for s in words:
            print("{0}: ({1}, {2}), ({3}, {4})".format(s[0], s[1], s[2], s[3], s[4]))
            if(args["save"]):
                cv2.imwrite("text{0}.jpg".format(processed), img[s[2]:s[4], s[1]:s[3]])
            if (args["debug"] > 1):
                cv2.rectangle(img,(s[1], s[2]),(s[3],s[4]),(0,255,0),1)
        if (args["debug"] > 1):
            cv2.imshow("img", img)
            cv2.waitKey(0)
    processed += 1
    if (processed > args["max"]):
        break

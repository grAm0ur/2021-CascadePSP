import numpy as np
from PIL import Image
import progressbar
import tqdm
from util.compute_boundary_acc import compute_boundary_acc_multi_class
from util.file_buffer import FileBuffer
from dataset.make_bb_trans import get_bb_position, scale_bb_by

from argparse import ArgumentParser
import glob
import os
import re
from pathlib import Path
from shutil import copyfile

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)
    
    return intersection, union 

if __name__ == "__main__":
    seg_dir = "Potsdam512/test_FastFCN"
    gt_dir = "Potsdam512/test_gt_L"
    mask_dir = "Output/finetune_full_2step"
    split_dir = mask_dir
    total_old_correct_pixels = 0
    total_new_correct_pixels = 0
    total_num_pixels = 0

    total_seg_mba = 0
    total_mask_mba = 0
    total_num_images = 0

    small_objects = 0

    num_classes = 6

    new_class_i = [0] * num_classes
    new_class_u = [0] * num_classes
    old_class_i = [0] * num_classes
    old_class_u = [0] * num_classes
    edge_class_pixel = [0] * num_classes
    old_gd_class_pixel = [0] * num_classes
    new_gd_class_pixel = [0] * num_classes
    error_data = []

    all_gts = os.listdir(seg_dir)
    mask_path = Path(mask_dir)

    i = 0
    for gt_name in tqdm.tqdm(all_gts):
        if i > 1:
         break
        i += 1
        print(gt_name)
        gt = np.array(Image.open(os.path.join(gt_dir, gt_name)
                                ).convert('L'))
        seg = np.array(Image.open(os.path.join(seg_dir, gt_name)
                                ).convert('L'))
        gt[gt==29] = 1
        gt[gt==76] = 5
        gt[gt==150] = 3
        gt[gt==179] = 2
        gt[gt==226] = 4
        gt[gt==255] = 0
        seg[seg==29] = 1
        seg[seg==76] = 5
        seg[seg==150] = 3
        seg[seg==179] = 2
        seg[seg==226] = 4
        seg[seg==255] = 0
    # We pick the highest confidence class label for overlapping region
        mask = seg.copy()
        confidence = np.zeros_like(gt) + 0.5
        keep = False
        for mask_name in mask_path.glob(gt_name[:-4] + '*mask*'):
            class_mask_prob = np.array(Image.open(mask_name).convert('L')).astype('float') / 255
            print(class_mask_prob)
            print(np.unique(class_mask_prob))
            class_string = re.search(r'\d', mask_name.name[::-1]).group()[::-1]
            this_class = int(class_string)
            print(this_class)
            class_seg = np.array(
                Image.open(
                    os.path.join(split_dir, mask_name.name.replace('mask', 'seg'))
                ).convert('L')
            ).astype('float') / 255
            try:
                rmin, rmax, cmin, cmax = get_bb_position(class_seg)
                bb = Image.getbbox(Image.fromarray(class_seg))
                print(bb)
                print(rmin, rmax, cmin, cmax)
                rmin, rmax, cmin, cmax = scale_bb_by(rmin, rmax, cmin, cmax, seg.shape[0], seg.shape[1], 0.25, 0.25)
            except:
            # Sometimes we cannot get a proper bounding box'''
                rmin = cmin = 0
                rmax, cmax = class_seg.shape

            if (cmax==cmin) or (rmax==rmin):
                print(gt_name, this_class)
                error_data.append(gt_name[:-4] + "_" + str(this_class))
                continue
            class_mask_prob = np.array(Image.fromarray(class_mask_prob).resize((cmax-cmin, rmax-rmin), Image.BILINEAR))
            im = Image.fromarray(class_mask_prob*255)
            im.show()
            background_classes = [0, 2]
            if this_class in background_classes:
                class_mask_prob = class_mask_prob * 0.51
                print("change to", class_mask_prob)
        # Record the current higher confidence level for each pixel
            mask[rmin:rmax, cmin:cmax] = np.where(class_mask_prob>confidence[rmin:rmax, cmin:cmax],
                                                this_class, mask[rmin:rmax, cmin:cmax])
            print(mask)
            confidence[rmin:rmax, cmin:cmax] = np.maximum(confidence[rmin:rmax, cmin:cmax], class_mask_prob)
            print(confidence)
        total_classes = np.union1d(np.unique(gt), np.unique(seg))
    # seg[gt==0] = 0
    # mask[gt==0] = 0
        '''total_classes = total_classes[1:] # Remove background class
        # Shift background class to -1
        total_classes -= 1'''
        idx = 0
        '''for c in total_classes:
            gt_class = (gt == c)
            seg_class = (seg == c)
            mask_class = (mask == c)
        
            old_i, old_u = get_iu(gt_class, seg_class)
            new_i, new_u = get_iu(gt_class, mask_class)

            total_old_correct_pixels += old_i
            total_new_correct_pixels += new_i
            total_num_pixels += gt_class.sum()

            new_class_i[idx] += new_i
            new_class_u[idx] += new_u
            old_class_i[idx] += old_i
            old_class_u[idx] += old_u
            idx += 1
        seg_acc, mask_acc = compute_boundary_acc_multi_class(gt, seg, mask)
        total_seg_mba += seg_acc
        total_mask_mba += mask_acc
        total_num_images += 1'''


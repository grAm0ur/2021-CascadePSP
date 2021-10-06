from PIL import Image
from sklearn.metrics import jaccard_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from util.file_buffer import FileBuffer
from pathlib import Path


def kappa(cm):
    """计算kappa值系数"""
    pe_rows = np.sum(cm, axis=0)
    pe_cols = np.sum(cm, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(cm) / float(sum_total)
    return (po - pe) / (1 - pe)


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=35, fontsize=10)
    plt.yticks(xlocations, labels, fontsize=10)
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)


def plot_cm_(cm):
    labels = ['Impervious Surfaces', 'Building', 'Low Vegetation', 'Tree', 'Car', 'Clutter/Background']
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=200)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    # for x_val, y_val in zip(x.flatten(), y.flatten()):
    #     c = cm_normalized[y_val][x_val]
    #     if c > 0.01:
    #         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    for i in range(6):
        for j in range(6):
            c = cm_normalized[i][j]
            if c > 0.01:
                if j == np.argmax(cm_normalized[i]):
                    plt.text(i, j, "%0.2f" % (c,), color='white', fontsize=16, va='center', ha='center')
                else:
                    plt.text(j, i, "%0.2f" % (c,), color='red', fontsize=16, va='center', ha='center')
            else:
                plt.text(j, i, "0.0", color='red', fontsize=16, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.3)

    plot_confusion_matrix(cm_normalized, labels, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig("../data/cm/dlinknet1024.png")
    plt.show()


def IOU(pred, target, n_classes=None):
    if n_classes is None:
        n_classes = [255, 29, 178, 149, 225, 76]
        # 0, 38, 75, 113, 14, 52
        # 179 -> 178  226 -> 225  150 -> 149
    ious = []
    # ignore IOU for background class
    i = [0, 0, 0, 0, 0, 0]
    u = [0, 0, 0, 0, 0, 0]
    for cls in n_classes:
        pred_inds = pred == cls
        target_inds = target == cls
        # target_sum = target_inds.sum()
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        i[n_classes.index(cls)] += intersection
        u[n_classes.index(cls)] += union
        # If there is no ground truth，do not include in evaluation
    return i, u


def multiClassImg_eval(seg_path, gt_path, log_path):
    # seg_path 为不包含类信息的文件保存路径
    # gt_path 为要评估的eval_post输出路径, 评估mask和seg时记得更改110行的后缀
    file_buffer = FileBuffer(log_path)
    file_buffer.write("multi-class image evaluation")
    file_buffer.write("dataset: ", seg_path)
    all_seg = os.listdir(gt_path)
    mlb = MultiLabelBinarizer(classes=[255, 29, 178, 149, 226, 76])
    total_iou = []
    idx = 0
    total_cm = np.zeros((6, 6))
    total_correct_pixel = 0
    total_pixel = 0
    total_i = 0
    intersec = [0, 0, 0, 0, 0, 0, 0]
    unio = [0, 0, 0, 0, 0, 0]
    for seg_name in tqdm.tqdm(all_seg):
        '''if idx == 1:
            break
        idx += 1'''
        seg = np.array(Image.open(seg_path + "/" + seg_name[:-4] + ".png").convert("L")).flatten()
        gt = np.array(Image.open(gt_path+"/"+seg_name[:-4] + ".png").convert("L")).flatten()
        seg[seg == 179] = 178
        seg[seg == 150] = 149
        gt[gt == 179] = 178
        gt[gt == 150] = 149
        seg[seg == 226] = 225
        gt[gt == 226] = 225
        '''seg = np.array(Image.open(seg_path + "/" + seg_name).convert("L")).flatten()
        gt = np.array(Image.open(gt_path + "/" + seg_name).convert("L")).flatten()'''
        # seg = mlb.fit_transform(seg)
        # gt = mlb.fit_transform(gt)
        iou = jaccard_score(gt, seg, average=None)
        total_correct_pixel += accuracy_score(gt, seg, normalize=False)
        total_pixel += len(gt)
        total_cm += confusion_matrix(gt, seg, labels=[255, 29, 178, 149, 225, 76])
        total_iou.append(iou)
        newi, newu = IOU(seg, gt)
        for i in range(6):
            intersec[i] += newi[i]
            unio[i] += newu[i]
    miou = []
    for i in range(6):
        miou.append(intersec[i]/unio[i])
    total_kappa = kappa(total_cm)
    file_buffer.write("Kappa:", total_kappa)
    file_buffer.write("Class IoU: ", miou)
    file_buffer.write("mIoU:", np.mean(miou))
    file_buffer.write("OA:", total_correct_pixel/total_pixel)
    file_buffer.write("total correct pixel/total pixel:", total_correct_pixel, "/", total_pixel)
    plot_cm_(total_cm)


def singleClassImg_eval(file_path, log_path, suffix="_mask.png"):
    file_buffer = FileBuffer(log_path)
    file_buffer.write("single-class image evaluation")
    file_buffer.write("dataset: ", file_path)
    gt_list = []
    im_list = []
    intersection = [0, 0, 0, 0, 0, 0]
    union = [0, 0, 0, 0, 0, 0]
    total_pixel = 0
    total_correct_pixel = 0
    total_cm = np.zeros((6, 6))
    for file in os.listdir(file_path):
        if "_gt.png" in file:
            gt_list.append(file[:-7])
            im_list.append(file[:-9])
    im_list = np.unique(im_list)
    for gt_name in tqdm.tqdm(gt_list):
        this_class = int(gt_name[-1])
        gt = np.array(Image.open(file_path + "/" + gt_name + "_gt.png").convert("L"))
        mask = np.array(Image.open(file_path + "/" + gt_name + suffix).convert("L"))
        mask[mask != 0] = 1

        new_i = np.count_nonzero(gt & mask)
        new_u = np.count_nonzero(gt | mask)
        intersection[this_class] += new_i
        union[this_class] += new_u
        total_correct_pixel += accuracy_score(gt.flatten(), mask.flatten(), normalize=False)
        total_pixel += len(gt.flatten())
    miou = []
    for i in range(6):
        miou.append(intersection[i]/union[i])
    file_buffer.write("Class IoU: ", miou)
    file_buffer.write("mIoU:", np.mean(miou))
    file_buffer.write("OA:", total_correct_pixel / total_pixel)
    file_buffer.write("total correct pixel/total pixel:", total_correct_pixel, "/", total_pixel)


if __name__ == "__main__":
    file_buffer = "../data/Output/results_post.txt"
    # multiClassImg_eval("Potsdam512/test_FastFCN", "Output/maskFusion/origin_full_2step_p3", file_buffer)
    multiClassImg_eval("../data/DLinknet/seg_6000", "../data/Larger_Potsdam/6000/gt", file_buffer)
    # print(np.unique(np.array(Image.open("../data/Output/DenseCRF/dlinknet1024/ top_potsdam_2_13_rgb-4.png").convert("L"))))

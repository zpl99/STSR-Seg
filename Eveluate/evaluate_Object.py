# -*- coding: utf-8 -*-
"""
@ Time    : 2021/1/24 11:28
@ Author  : Xu Penglei
@ Email   : xupenglei87@163.com
@ File    : obj_level_evaluate.py
@ Desc    : 求取对象级别的评价
"""
import numpy as np
import skimage.io as io
from skimage import measure
from tqdm import tqdm
import os
import csv
import cv2
import csv
from glob import glob
import matplotlib.pyplot as plt


def obj_weighted_statistics(GT_labels, GT_region, pred):
    """通过面积与相交数进行加权，求recall和pre"""
    record = []
    for subR in GT_region:
        label = subR['label']
        bbox = subR['bbox']
        sub_GT = (GT_labels[bbox[0]:bbox[2], bbox[1]:bbox[3]] == label) * 1
        sub_pred = pred[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        inter = sub_GT * sub_pred
        inter_area = inter.sum()
        inter_num_pred = len(measure.regionprops(measure.label(inter, connectivity=2)))
        if inter_num_pred > 0:
            criterion = (inter_area / subR['area']) / inter_num_pred
        else:
            criterion = 0
        record.append(criterion)
    if len(record) == 0:
        return 0, []
    else:
        return np.mean(record), record


def count_regions_in_regions(region1, label1, label2):
    """在region1中有几个label2的部分"""
    label = region1['label']
    bbox = region1['bbox']
    sub_region1 = (label1[bbox[0]:bbox[2], bbox[1]:bbox[3]] == label) * 1
    sub_label2 = label2[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    inter_label2 = sub_region1 * sub_label2
    inter_region = np.where(inter_label2 > 1, 1, inter_label2)
    uniq_val = np.unique(inter_label2)
    uq_list = list(uniq_val)
    if 0 in uq_list:
        uq_list.remove(0)
    return uq_list, inter_region.sum(), sub_region1.sum()


def obj_iou(GT_labels, GT_region, pred_labels, pred_region):
    """自定义的对象级IOU，iou/(所交pred的数目*所交pred内部包含GT的数目)"""
    record = []
    for subR in GT_region:
        regions_in_subR, inter_area, subR_area = count_regions_in_regions(subR, GT_labels, pred_labels)
        if inter_area == 0:
            record.append(0)
        else:
            n_regions_in_pred = 0
            tot_pred_area = 0
            for pred_label in regions_in_subR:
                pred_sub_region = pred_region[pred_label - 1]
                regions_in_pred_subR, _, pred_subR_area = count_regions_in_regions(pred_sub_region, pred_labels,
                                                                                   GT_labels)
                n_regions_in_pred += len(regions_in_pred_subR)
                tot_pred_area += pred_subR_area
            iou = (inter_area / (subR_area + tot_pred_area - inter_area)) / (len(regions_in_subR) * n_regions_in_pred)
            record.append(iou)
    return np.mean(record)


def individual_stat(GT, GT_labels, GT_regions, pred, pred_labels, pred_regions):
    m_recall, recall_records = obj_weighted_statistics(GT_labels, GT_regions, pred)
    m_precision, precision_records = obj_weighted_statistics(pred_labels, pred_regions, GT)
    if m_precision + m_recall < 1e-8:
        m_f1 = 0
    else:
        m_f1 = 2 * m_precision * m_recall / (m_precision + m_recall)
    return m_recall, m_precision, m_f1, recall_records, precision_records


def padding_list(ls, transpose=True):
    """不等长list用None填充为等长"""
    max_len = np.max([len(n) for n in ls])
    pad_n = [max_len - len(n) for n in ls]
    new_ls = []
    for i, l in enumerate(ls):
        new_ls.append(l + [None] * int(pad_n[i]))
    if transpose:
        new_ls_T = [[n[i] for n in new_ls] for i in range(max_len)]
        return new_ls_T
    else:
        return new_ls


if __name__ == '__main__':
    from fnmatch import fnmatch
    # GT_dir = r'Z:\Data\dataset\hr\label\test'
    # pre_dir = r'E:\哨兵2号实验\pred\SS'
    GT_dir = r'F:\Data\各数据集小测试\Sentinel2_Toy\label_test'
    pre_dir = r'F:\ExperimentsResult\哨兵2号实验\pred_extraData\SRSS_noExtra'
    names = os.listdir(pre_dir)
    # names = ['EDSR_NASUNET']
    pred_dir_list = [os.path.join(pre_dir, n) for n in names]
    # out_dir = r'E:\哨兵2号实验\eval\SS\obj'
    out_dir = r'F:\ExperimentsResult\哨兵2号实验\eval_extraData\SRSS_noExtra\obj'
    os.makedirs(out_dir,exist_ok=True)

    recall_csv = os.path.join(out_dir, 'recall.csv')
    precision_csv = os.path.join(out_dir, 'precesion.csv')
    f1_csv = os.path.join(out_dir, 'f1.csv')
    recall_csv_part = os.path.join(out_dir, 'recall_part.csv')
    precision_csv_part = os.path.join(out_dir, 'precesion_part.csv')
    stat_csv = os.path.join(out_dir, 'stat.csv')

    pred_n = len(pred_dir_list)
    m_recall = [[] for n in range(pred_n)]
    m_precision = [[] for n in range(pred_n)]
    m_f1 = [[] for n in range(pred_n)]
    m_recall_part = [[] for n in range(pred_n)]
    m_precision_part = [[] for n in range(pred_n)]
    names = [n for n in os.listdir(pred_dir_list[0]) if (fnmatch(n,'*.tif') or fnmatch(n, '*.tiff')) and '_.' not in n]
    for item in tqdm(names):
        GT_path = os.path.join(GT_dir, item.replace('.tiff','.tif'))
        GT = (io.imread(GT_path) == 255) * 1
        GT_label = measure.label(GT, connectivity=2)
        GT_region = measure.regionprops(GT_label)
        w, h = GT.shape[:2]
        for i, p in enumerate(pred_dir_list):
            try:
                pred_path = os.path.join(p, item)
                pred = (io.imread(pred_path) == 255) * 1
            except:
                pred_path = os.path.join(p, item.replace('.tif','.tiff'))
                pred = (io.imread(pred_path) == 255) * 1
            w_p, h_p = pred.shape[:2]
            if w_p != w or h_p != h:
                pred = cv2.resize(pred, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
            pred_label = measure.label(pred, connectivity=2)
            pred_region = measure.regionprops(pred_label)
            m_r, m_p, f1, m_r_record, m_p_record = individual_stat(GT, GT_label, GT_region, pred, pred_label,
                                                                   pred_region)
            m_recall[i].append(m_r)
            m_precision[i].append(m_p)
            m_f1[i].append(f1)
            m_recall_part[i].extend(m_r_record)
            m_precision_part[i].extend(m_p_record)
    print([np.mean(n) for n in m_f1])
    print([np.mean(n) for n in m_recall])
    print([np.mean(n) for n in m_precision])

    dirs = [os.path.basename(n) for n in pred_dir_list]

    f_recall = open(recall_csv, 'a', newline='')
    recall_writer = csv.writer(f_recall)
    recall_writer.writerow(dirs)
    recall_writer.writerows(np.array(m_recall).T.tolist())
    f_precision = open(precision_csv, 'a', newline='')
    precision_writer = csv.writer(f_precision)
    precision_writer.writerow(dirs)
    precision_writer.writerows(np.array(m_precision).T.tolist())
    f_f1 = open(f1_csv, 'a', newline='')
    f1_writer = csv.writer(f_f1)
    f1_writer.writerow(dirs)
    f1_writer.writerows(np.array(m_f1).T.tolist())
    f_recall.close()
    f_precision.close()
    f_f1.close()

    print('PART')
    print([np.mean(n) for n in m_recall_part])
    print([np.mean(n) for n in m_precision_part])

    f_recall_part = open(recall_csv_part, 'a', newline='')
    recall_part_writer = csv.writer(f_recall_part)
    recall_part_writer.writerow(dirs)
    recall_part_writer.writerows(np.array(m_recall_part).T.tolist())
    f_recall_part.close()

    f_precision_part = open(precision_csv_part, 'a', newline='')
    precision_part_writer = csv.writer(f_precision_part)
    precision_part_writer.writerow(dirs)
    new_m_precision_part = padding_list(m_precision_part, transpose=True)
    precision_part_writer.writerows(new_m_precision_part)
    f_precision_part.close()

    with open(stat_csv, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['img-wise'])
        csv_writer.writerow([' '] + dirs)
        csv_writer.writerow(['Recall'] + [np.mean(n) for n in m_recall])
        csv_writer.writerow(['Precision'] + [np.mean(n) for n in m_precision])
        csv_writer.writerow(['F1'] + [np.mean(n) for n in m_f1])

        csv_writer.writerow(['GT-object-wise'])
        csv_writer.writerow([' '] + dirs)
        mean_recall_part = [np.mean(n) for n in m_recall_part]
        mean_precision_part = [np.mean(n) for n in m_precision_part]
        mean_f1_part = 2 * np.array(mean_precision_part) * np.array(mean_recall_part) / (
                    np.array(mean_precision_part) + np.array(mean_recall_part))
        csv_writer.writerow(['Recall'] + mean_recall_part)
        csv_writer.writerow(['Precison'] + mean_precision_part)
        csv_writer.writerow(['F1'] + mean_f1_part.tolist())
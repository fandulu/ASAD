# Edit by Fan Yang @2021


import os
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from easydict import EasyDict as edict
from .io import  print_metrics
from .bbox import bbox_overlap
from .measurements import clear_mot_hungarian, idmeasures

def evaluate_sequence(trackDB, gtDB, distractor_ids, iou_thres=0.5):
    """
    Evaluate single sequence
    trackDB: tracking result data structure
    gtDB: ground-truth data structure
    iou_thres: bounding box overlap threshold

    """

    mme, c, fp, g, missed, d, M, allfps = clear_mot_hungarian(
        trackDB, gtDB, iou_thres)

    gt_frames = np.unique(gtDB[:, 0])
    gt_ids = np.unique(gtDB[:, 1])
    st_ids = np.unique(trackDB[:, 1])
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)

    FN = sum(missed)
    FP = sum(fp)
    IDS = sum(mme)
    # MOTP = sum(iou) / # corrected boxes
    MOTP = (sum(sum(d)) / sum(c)) * 100
    # MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
    MOTAL = (1 - (
        sum(fp) + sum(missed) + np.log10(sum(mme) + 1)) / sum(g)) * 100
    MOTA = (1 - (sum(fp) + sum(missed) + sum(mme)) / sum(g)) * \
        100                      # MOTA = 1 - (# fp + # fn + # ids) / # gts
    # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    recall = sum(c) / sum(g) * 100
    # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    precision = sum(c) / (sum(fp) + sum(c)) * 100
    # FAR = sum(fp) / # frames
    FAR = sum(fp) / f_gt
    MT_stats = np.zeros((n_gt, ), dtype=float)
    for i in range(n_gt):
        gt_in_person = np.where(gtDB[:, 1] == gt_ids[i])[0]
        gt_total_len = len(gt_in_person)
        gt_frames_tmp = gtDB[gt_in_person, 0].astype(int)
        gt_frames_list = list(gt_frames)
        st_total_len = sum(
            [1 if i in M[gt_frames_list.index(f)].keys() else 0
                for f in gt_frames_tmp])
        ratio = float(st_total_len) / gt_total_len

        if ratio < 0.2:
            MT_stats[i] = 1
        elif ratio >= 0.8:
            MT_stats[i] = 3
        else:
            MT_stats[i] = 2

    ML = len(np.where(MT_stats == 1)[0])
    PT = len(np.where(MT_stats == 2)[0])
    MT = len(np.where(MT_stats == 3)[0])

    # fragment
    fr = np.zeros((n_gt, ), dtype=int)
    M_arr = np.zeros((f_gt, n_gt), dtype=int)

    for i in range(f_gt):
        for gid in M[i].keys():
            M_arr[i, gid] = M[i][gid] + 1

    for i in range(n_gt):
        occur = np.where(M_arr[:, i] > 0)[0]
        occur = np.where(np.diff(occur) != 1)[0]
        fr[i] = len(occur)
    FRA = sum(fr)
    idmetrics = idmeasures(gtDB, trackDB, iou_thres)
    metrics = [idmetrics.IDF1, idmetrics.IDP, idmetrics.IDR, recall,
               precision, FAR, n_gt, MT, PT, ML, FP, FN, IDS, FRA,
               MOTA, MOTP, MOTAL]
    extra_info = edict()
    extra_info.mme = sum(mme)
    extra_info.c = sum(c)
    extra_info.fp = sum(fp)
    extra_info.g = sum(g)
    extra_info.missed = sum(missed)
    extra_info.d = d
    # extra_info.m = M
    extra_info.f_gt = f_gt
    extra_info.n_gt = n_gt
    extra_info.n_st = n_st
#    extra_info.allfps = allfps

    extra_info.ML = ML
    extra_info.PT = PT
    extra_info.MT = MT
    extra_info.FRA = FRA
    extra_info.idmetrics = idmetrics
    return metrics, extra_info


def evaluate_bm(all_metrics):
    """
    Evaluate whole benchmark, summaries all metrics
    """
    f_gt, n_gt, n_st = 0, 0, 0
    nbox_gt, nbox_st = 0, 0
    c, g, fp, missed, ids = 0, 0, 0, 0, 0
    IDTP, IDFP, IDFN = 0, 0, 0
    MT, ML, PT, FRA = 0, 0, 0, 0
    overlap_sum = 0
    for i in range(len(all_metrics)):
        nbox_gt += all_metrics[i].idmetrics.nbox_gt
        nbox_st += all_metrics[i].idmetrics.nbox_st
        # Total ID Measures
        IDTP += all_metrics[i].idmetrics.IDTP
        IDFP += all_metrics[i].idmetrics.IDFP
        IDFN += all_metrics[i].idmetrics.IDFN
        # Total ID Measures
        MT += all_metrics[i].MT
        ML += all_metrics[i].ML
        missed += all_metrics[i].missed
        ids += all_metrics[i].mme
        overlap_sum += sum(sum(all_metrics[i].d))
        
    # IDP = IDTP / (IDTP + IDFP)
    IDP = IDTP / (IDTP + IDFP) * 100
    IDR = IDTP / (IDTP + IDFN) * 100
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100
  
    metrics = [IDF1, MT, ML, ids]
    return metrics
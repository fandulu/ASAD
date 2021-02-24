from utils.iou import IOU_one2one
from scipy.optimize import linear_sum_assignment
import numpy as np

def matching_boxes(true_boxes, pred_boxes):
    N_1 = len(true_boxes)
    N_2 = len(pred_boxes)
    iou_mat = np.zeros([N_1,N_2])  
    for i in range(N_1):                    
        for j in range(N_2):
            p_b = true_boxes[j]
            iou_mat[i,j] = IOU_one2one(true_boxes[i], pred_boxes[j])

    iou_mat = 1-iou_mat

    # set IoU threshold 0.5
    iou_mat[iou_mat>0.5] = 1

    # linear assignment
    row_ind, col_ind = linear_sum_assignment(iou_mat)

    # remove IoU>0.5 matching
    row_ind_new = row_ind[iou_mat[row_ind, col_ind]<1] 
    col_ind_new = col_ind[iou_mat[row_ind, col_ind]<1] 
    row_ind = row_ind_new
    col_ind = col_ind_new

    matches = []
    for row, col in zip(row_ind,col_ind):
        matches.append((row, col))
        
    return matches
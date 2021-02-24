import numpy as np
import cv2
import pandas as pd
import random
import argparse
import pickle as pkl
import csv
import glob
import json
import pickle
import copy
from utils import coco_tools
from utils.evaluate_MOT import *
from utils.blockPrint import blockPrint,enablePrint
from utils.matching_boxes import matching_boxes
from sklearn.metrics import hamming_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="A-AVA/action_pred_val.pickle", help="The prediction file")
    parser.add_argument("--true_file", type=str, default="A-AVA/action_anno_val.pickle", help="The ground-truth file")
    args = parser.parse_args()
    return args

def main(args):
    
    with open(args.pred_file, 'rb') as handle:
        ann_ava_pred = pickle.load(handle)
    with open(args.true_file, 'rb') as handle:
        ann_ava_true = pickle.load(handle)
        
    # Preprocess data 
    data_true = {} 
    pid_list = []
    for video_id in list(ann_ava_true.keys()):
        data_true[video_id] = []

        key_img_list = list(ann_ava_true[video_id]['obj_info'].keys())
        for key_img_id in key_img_list:  
            key_frame = ann_ava_true[video_id]['img_info'][key_img_id]['frame']    
            for pid in ann_ava_true[video_id]['obj_info'][key_img_id].keys():        
                x,y,w,h = np.array(ann_ava_true[video_id]['obj_info'][key_img_id][pid]['bbox'])
                x1,y1,x2,y2 = x,y,x+w,y+h
                data_true[video_id].append([key_frame,pid,x1,y1,x2,y2,1,0,1])
                pid_list.append('{}_{}'.format(video_id,pid))
        data_true[video_id] = np.array(data_true[video_id])
    num_pid = len(np.unique(pid_list))
        
    data_pred = {}
    for video_id in list(ann_ava_pred.keys()):
        data_pred[video_id] = []

        key_img_list = list(ann_ava_pred[video_id]['obj_info'].keys())
        for key_img_id in key_img_list:  
            key_frame = ann_ava_pred[video_id]['img_info'][key_img_id]['frame']    
            for pid in ann_ava_pred[video_id]['obj_info'][key_img_id].keys():        
                x,y,w,h = np.array(ann_ava_pred[video_id]['obj_info'][key_img_id][pid]['bbox'])
                x1,y1,x2,y2 = x,y,x+w,y+h
                data_pred[video_id].append([key_frame,pid,x1,y1,x2,y2,1,0,1])
        data_pred[video_id] = np.array(data_pred[video_id])
    
    
    # Evaluate MOT
    all_info = []
    for video_id in list(data_true.keys()):

        trackDB = data_pred[video_id]
        gtDB =  data_true[video_id]
        metrics, extra_info = evaluate_sequence(trackDB, gtDB, distractor_ids=[])
        all_info.append(extra_info)

    all_metrics = np.array(evaluate_bm(all_info))  
    all_metrics[1:3] = all_metrics[1:3]/num_pid*100
    
    # Evaluate Detection
    image_ids_det = []
    image_ids_gt = []
    gt_boxes = []
    gt_classes = []
    det_boxes = []
    det_classes = []
    det_scores = []
    
    
    for video_id in list(ann_ava_true.keys()):
        key_img_list = list(ann_ava_true[video_id]['obj_info'].keys())
        for key_img_id in key_img_list:  
            key_frame = ann_ava_true[video_id]['img_info'][key_img_id]['frame'] 
            gt_b = []
            gt_c = []
            for pid in ann_ava_true[video_id]['obj_info'][key_img_id].keys():        
                x,y,w,h = np.array(ann_ava_true[video_id]['obj_info'][key_img_id][pid]['bbox'])
                x1,y1,x2,y2 = x,y,x+w,y+h
                gt_b.append(np.array([x1,y1,x2,y2]))
                gt_c.append(0)
            image_ids_gt.append('{}_{}'.format(video_id,key_frame))
            gt_boxes.append(np.array(gt_b))
            gt_classes.append(np.array(gt_c))
            
    for video_id in list(ann_ava_pred.keys()):
        key_img_list = list(ann_ava_pred[video_id]['obj_info'].keys())
        for key_img_id in key_img_list:  
            key_frame = ann_ava_pred[video_id]['img_info'][key_img_id]['frame'] 
            det_b = []
            det_c = []
            det_s = []
            for pid in ann_ava_pred[video_id]['obj_info'][key_img_id].keys():        
                x,y,w,h = np.array(ann_ava_pred[video_id]['obj_info'][key_img_id][pid]['bbox'])
                x1,y1,x2,y2 = x,y,x+w,y+h
                det_b.append(np.array([x1,y1,x2,y2]))
                det_c.append(0)
                det_s.append(1.0)
            image_ids_det.append('{}_{}'.format(video_id,key_frame))
            det_boxes.append(np.array(det_b))
            det_classes.append(np.array(det_c))
            det_scores.append(np.array(det_s))
            
    #Convert ground truth list to dict
    categories = np.array([{'id': 0, 'name': 'fake'}])

    groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
          image_ids_gt, gt_boxes, gt_classes,
          categories, output_path=None)

    #Convert detections list to dict
    detections_list = coco_tools.ExportDetectionsToCOCO(
        image_ids_det, det_boxes, det_scores,
        det_classes, categories, output_path=None)

    groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    detections = groundtruth.LoadAnnotations(detections_list)
    evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,agnostic_mode=False)

    blockPrint()
    det_metrics, _ = evaluator.ComputeMetrics()
    
    
    HL_all = []
    for video_id in list(ann_ava_true.keys()):
        data_true[video_id] = []
        key_img_list = list(ann_ava_true[video_id]['obj_info'].keys())
        for key_img_id in key_img_list:  
            key_frame = ann_ava_true[video_id]['img_info'][key_img_id]['frame']  

            true_boxes = []
            true_actions = []
            for pid in ann_ava_true[video_id]['obj_info'][key_img_id].keys():        
                x,y,w,h = ann_ava_true[video_id]['obj_info'][key_img_id][pid]['bbox']
                x1,y1,x2,y2 = x,y,x+w,y+h
                true_boxes.append([x1,y1,x2,y2])
                actions = np.zeros(81)
                actions[ann_ava_true[video_id]['obj_info'][key_img_id][pid]['action']] = 1
                actions = actions[:80]
                true_actions.append(actions)

            pred_boxes = [] 
            pred_actions = []
            for pid in ann_ava_pred[video_id]['obj_info'][key_img_id].keys():        
                x,y,w,h = ann_ava_pred[video_id]['obj_info'][key_img_id][pid]['bbox']
                x1,y1,x2,y2 = x,y,x+w,y+h
                pred_boxes.append([x1,y1,x2,y2])
                actions = np.zeros(81)
                actions[ann_ava_pred[video_id]['obj_info'][key_img_id][pid]['action']] = 1
                actions = actions[:80]
                pred_actions.append(actions)


            true_boxes = np.stack(true_boxes)  
            pred_boxes = np.stack(pred_boxes)

            matches = matching_boxes(true_boxes, pred_boxes)

            for pair in matches:
                y_true = true_actions[pair[0]]
                y_pred = pred_actions[pair[1]]
                hl = hamming_loss(y_true, y_pred)

                HL_all.append(hl)

    HL_avg = np.mean(HL_all)
    
    enablePrint()
    print("mAP@.50IoU:{},  IDF1:{},   MT:{}, ML:{}, ID s.w.:{},   HL:{}".format(det_metrics['Precision/mAP@.50IOU'], 
                                                                                all_metrics[0],
                                                                                all_metrics[1],
                                                                                all_metrics[2],
                                                                                all_metrics[3],                                                                                                                   HL_avg))


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)

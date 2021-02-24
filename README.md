# Actor-identified Spatiotemporal Action Detection (ASAD)

## Datasets

### A-AVA Dataset Download

Our A-AVA dataset is extended from the [AVA dataset] (http://research.google.com/ava/) and the [TAO dataset] (https://taodataset.org/). For more details and copyrights, please refer to their webpages.

You can download our A-AVA dataset from 
[Download](https://drive.google.com/file/d/19p84A5rZUGtExpWnwLVc7n4sgypvITIt/view?usp=sharing)

### A-AVA Dataset Structure

The A-AVA dataset has the following structures:
```
data_val = {1230(video id):
                {'video_info': 
                                {'video_id': 1230,
                                'width': 1280,
                                'height': 720,
                                'video_path': 'val/AVA/keUOiCcHtoQ_scene_28_124948-125707'
                                },
                    'img_info':
                                {47560 (image id):  
                                    {'img_path': 'val/AVA/keUOiCcHtoQ_scene_28_124948-125707/frame0051.jpg',
                                    'frame': 50},
                                ...
                                }
                    'obj_info':
                                {47560 (image id): 
                                    {0 (object id): 
                                        {'bbox': (154, 79, 341, 281), 
                                        'image_id': 47560, 
                                        'action': [14]},
                                    ...}
                                ...}
                }
        ...
        }
```

To do the evaluation, the output should have the same format as the above example.

## Quantitative evaluation

We provide the evaluation scripts. Which can be run by

```
python run_evaluation.py \
--pred_file A-AVA/action_pred_val.pickle \
--true_file A-AVA/action_anno_val.pickle
```

The results will be illustrated as
```
mAP@.50IoU: xxx,  IDF1: xxx,   MT: xxx, ML: xxx, ID s.w.: xxx,   HL: xxx
```

<a name="license"></a>
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

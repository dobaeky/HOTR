# ------------------------------------------------------------------------
# HOTR official code : hotr/engine/evaluator_vcoco.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import torch
import time
import datetime
import numpy as np
import hotr.util.misc as utils
import hotr.util.logger as loggers
from hotr.data.evaluators.vcoco_eval import VCocoEvaluator
from hotr.util.box_ops import rescale_bboxes, rescale_pairs
from PIL import Image, ImageDraw, ImageFont

import cv2
from matplotlib import pyplot as plt
import wandb
import json
@torch.no_grad()
def vcoco_evaluate(model, criterion, postprocessors, data_loader, device, output_dir, thr):
    coco_cls=['No','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','12','stop sign','parking meter','bench','bird','cat','dog','horse','sheep',
    'cow','elephant','bear','zebra','giraffe','26','backpack','umbrella','29','30',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','45','wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
    'cake','chair','couch','potted plant','bed','66','dining table','68','69','toilet',
    '71','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
    'sink','refrigerator','83','book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
    '91']
    vcoco_cls= ['hold', 'stand', 'sit', 'ride', 'walk',
     'look', 'hit_', 'hit', 'eat', 'eat_', 
     'jump', 'lay', 'talk_on_phone', 'carry', 'throw',
      'catch', 'cut_', 'cut', 'run', 'work_on_computer', 
      'ski', 'surf', 'skateboard', 'smile', 'drink',
       'kick', 'point', 'read', 'snowboard']

    model.eval()
    criterion.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (V-COCO)'
    ##print('----data_loader len----')
    ##print(len(data_loader))
    #print(type(data_loader)) #<class 'torch.utils.data.dataloader.DataLoader'>
    print_freq = 1 # len(data_loader)
    res = {}
    hoi_recognition_time = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = utils.reduce_dict(loss_dict) # ddp gathering

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='vcoco')
        targets = process_target(targets, orig_target_sizes)


        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        res.update(
            {target['image_id'].item():\
                {'target': target, 'prediction': output} for target, output in zip(targets, results)
            }
        )

    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")
    print('result is next line')

    base_img_path="v-coco/coco/images/val2014/COCO_val2014_000000"
    base_img_output="results_imgs/"
    list_keys=list(res.keys())
    carry_set=set()
    skate_set=set()
    for img_name in list_keys:
        if len(str(img_name))==2:
            img_path=base_img_path+'0000'+str(img_name)+'.jpg'
        elif len(str(img_name))==3:
            img_path=base_img_path+'000'+str(img_name)+'.jpg'
        elif len(str(img_name))==4:
            img_path=base_img_path+'00'+str(img_name)+'.jpg'
        elif len(str(img_name))==5:
            img_path=base_img_path+'0'+str(img_name)+'.jpg'
        elif len(str(img_name))==6:
            img_path=base_img_path+str(img_name)+'.jpg'
        else:
            print("why bloody hell long name")

       
    
        img_a = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img_a)
        t_i=1
        font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf",22)
        for i in range(len(res[img_name]['target']['labels'])): 
        
            text_pos=int(res[img_name]['target']['boxes'][i,0])+3,int(res[img_name]['target']['boxes'][i,1])#object text pos
            if res[img_name]['target']['labels'][i]==1:
                draw.text(text_pos,coco_cls[res[img_name]['target']['labels'][i]],(0,255,0),font=font)#person bbox
                draw.rectangle((int(res[img_name]['target']['boxes'][i,0]),int(res[img_name]['target']['boxes'][i,1]),int(res[img_name]['target']['boxes'][i,2]),int(res[img_name]['target']['boxes'][i,3])),outline=(0,255,0), width=3)
            else:
                draw.text(text_pos,coco_cls[res[img_name]['target']['labels'][i]],(50,150,50),font=font) #object bbox
                draw.rectangle((int(res[img_name]['target']['boxes'][i,0]),int(res[img_name]['target']['boxes'][i,1]),int(res[img_name]['target']['boxes'][i,2]),int(res[img_name]['target']['boxes'][i,3])),outline=(50,150,50), width=3)
        
        
        #print(torch.where(res[26624]['target']['pair_targets']>0)) # (tensor([1, 3], device='cuda:0'),)
        if res[img_name]['target']['pair_targets'].size(dim=0) == 1 and res[img_name]['target']['pair_targets'][0]==-1:
            for i in torch.where(res[img_name]['target']['pair_actions'][i]>0)[0]:
                txt_pos=(0,int(res[img_name]['target']['orig_size'][0]-22*t_i))
                draw.text(txt_pos,'person '+vcoco_cls[int(i)],(255,0,255),font=font)
                t_i+=1
        else:
            for i in torch.where(res[img_name]['target']['pair_targets']>0)[0]: #es[26624]['target']['pair_targets']>0)[0] = tensor([1,3])
                
                #print(torch.where(res[img_name]['target']['pair_actions'][i]>0))
                #= [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                # 0, 0, 0, 0, 0] -> so (tensor([5]),)
                for j in torch.where(res[img_name]['target']['pair_actions'][i]>0)[0]: #i=0,2 
                #i=1 ) j=res[26624][]... = 5
                    #print(res[26624]['target']['orig_size'])
                    #print(res[26624]['target']['orig_size'][0])
                    #print(res[26624]['target']['orig_size'][1])
                    txt_pos=(0,int(res[img_name]['target']['orig_size'][0]-22*t_i))
                    #print(txt_pos)
                    draw.text(txt_pos,'person '+vcoco_cls[j]+' '+coco_cls[res[img_name]['target']['pair_targets'][i]],(0,0,255),font=font)
                    if vcoco_cls[j]=='carry':
                        carry_set.add(img_name)
                    elif vcoco_cls[j]=='skateboard':
                        skate_set.add(img_name)
                    t_i+=1
        t_i=0

        for i in range(29):
            for j in range(res[img_name]['prediction']['h_box'].size()[0]):

                if torch.where(res[img_name]['prediction']['pair_score'][i,j]>=1)[0].size(dim=0)>0 and \
                torch.where(res[img_name]['prediction']['pair_score'][i,j]>=1)[0][0]!=100 and i!=1 and i!=4 \
                and i!=18 and i!=23 and i!=26:

                    a_idx=i
                    h_idx=j
                    text_pos=int(res[img_name]['prediction']['h_box'][h_idx][0])+3,int(res[img_name]['prediction']['h_box'][h_idx][1])
                    draw.text(text_pos,'person',(255,128,0),font=font)
                    draw.rectangle((int(res[img_name]['prediction']['h_box'][h_idx][0]),int(res[img_name]['prediction']['h_box'][h_idx][1]),int(res[img_name]['prediction']['h_box'][h_idx][2]),int(res[img_name]['prediction']['h_box'][h_idx][3])),outline=(255,128,0), width=3)


                    for k in torch.where(res[img_name]['prediction']['pair_score'][i,j]>=1)[0]:
                        if int(k)!=100:
                            text_pos=int(res[img_name]['prediction']['o_box'][k][0])+3,int(res[img_name]['prediction']['o_box'][k][1])
                            draw.text(text_pos,coco_cls[res[img_name]['prediction']['labels'][k]],(255,255,0),font=font)
                            draw.rectangle((int(res[img_name]['prediction']['o_box'][k][0]),int(res[img_name]['prediction']['o_box'][k][1]),int(res[img_name]['prediction']['o_box'][k][2]),int(res[img_name]['prediction']['o_box'][k][3])),outline=(255,255,0), width=3)
                            txt_pos=(0,int(22*t_i))
                            draw.text(txt_pos,'person '+vcoco_cls[a_idx]+' '+coco_cls[res[img_name]['prediction']['labels'][k]],(255,0,0),font=font)
                            t_i+=1

        img_output_path = base_img_output+str(img_name)+'gp.jpg'
        img_a.save(img_output_path)
        print(img_name)

    #print(len(carry_set))
    #print(len(skate_set))
    #print(carry_set)
    #print(skate_set)
    file_carry=open('carry_ids.txt','w')
    file_skate=open('skate_ids.txt','w')
    file_skate.write('skate_ids\n')
    file_carry.write('carry ids\n')
    for i in carry_set:
        file_carry.write(str(i))
        file_carry.write('\n')
    file_carry.close()
    for i in skate_set:
        file_skate.write(str(i))
        file_skate.write('\n')
    file_skate.close()

    start_time = time.time()
    gather_res = utils.all_gather(res)
    total_res= {}
    for dist_res in gather_res:
        total_res.update(dist_res)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"[stats] Distributed Gathering Time : {total_time_str}")

    return total_res

def vcoco_accumulate(total_res, args, print_results, wandb_log):
    vcoco_evaluator = VCocoEvaluator(args)
    vcoco_evaluator.update(total_res)

    print(f"[stats] Score Matrix Generation completed!!          ")

    scenario1 = vcoco_evaluator.role_eval1.evaluate(print_results)
    scenario2 = vcoco_evaluator.role_eval2.evaluate(print_results)

    if wandb_log:
        wandb.log({
            'scenario1': scenario1,
            'scenario2': scenario2
        })

    return scenario1, scenario2

def process_target(targets, target_sizes):
    for idx, (target, target_size) in enumerate(zip(targets, target_sizes)):
        labels = target['labels']
        valid_boxes_inds = (labels > 0)

        targets[idx]['boxes'] = rescale_bboxes(target['boxes'], target_size) # boxes
        targets[idx]['pair_boxes'] = rescale_pairs(target['pair_boxes'], target_size) # pairs

    return targets

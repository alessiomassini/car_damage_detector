import os
import shutil
import time
import json
import argparse
import numpy as np
import cv2
from PIL import Image



weights = ['cardamage_final.weights']

img_dir = '../test_images'

################ ------  DEFINE THE FUNCTIONS -----  ###############


def get_coords(json_file, filename):
    
    image = Image.open(img_dir + '/' + filename)
    width, height = image.size

    def rescale_x(x):
        if x < 0: x = 0
        elif x > 1: x = 1
        return int(x*width)

    def rescale_y(x):
        if x < 0: x = 0
        elif x > 1: x = 1
        return int(x*height)

    with open(json_file, 'r') as f:
        data = json.load(f)[0]['objects']

    detections = []
    for i in data:
        xmin = (i['relative_coordinates']['center_x'] - i['relative_coordinates']['width'] / 2)
        xmax = (xmin + i['relative_coordinates']['width'])
        ymin = i['relative_coordinates']['center_y'] - i['relative_coordinates']['height'] / 2
        ymax = ymin + i['relative_coordinates']['height']
    
        detections.append([i['name'], rescale_x(xmin),
                                      rescale_y(ymin),
                                      rescale_x(xmax),
                                      rescale_y(ymax), i['confidence']])
    
    return detections


def non_max_suppression(detections_, tolerance=0.45):

    # If no bounding boxes, return empty list
    if len(detections_) == 0:
        return []
    
    bounding_boxes=[]
    confidence_score=[]
 
    for l in detections_:
        bounding_boxes.append( l[1:5] )
        confidence_score.append(l[5])      
    
    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        
        left = np.where(ratio < tolerance)
        order = order[left]
    
    output_labels=[]
        
    for j in range(len(picked_boxes)):
        for i in range(len(detections_)):    
            if(picked_boxes[j]==bounding_boxes[i] and picked_score[j]==detections_[i][5]):
                output_labels.append(detections_[i][0])    
    
    mask=[True]*len(output_labels)

    for i in range(len(output_labels)):
        for j in range(i+1, len(output_labels)):
            if output_labels[i]==output_labels[j]:
                #print(output_labels[i])
                areai=(picked_boxes[i][2]-picked_boxes[i][0])*(picked_boxes[i][3]-picked_boxes[i][1])
                areaj=(picked_boxes[j][2]-picked_boxes[j][0])*(picked_boxes[j][3]-picked_boxes[j][1])
                #print(areai, areaj)
                areamin=min(areai,areaj)
                
                if areamin == areai:
                    toremove=i
                else:
                    toremove=j

                #compute intersection area:
                x1 = max(picked_boxes[i][0], picked_boxes[j][0])
                x2 = min(picked_boxes[i][2], picked_boxes[j][2])
                y1 = max(picked_boxes[i][1], picked_boxes[j][1])
                y2 = min(picked_boxes[i][3], picked_boxes[j][3])
                #print(x1,x2,y1,y2)

                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                intersection = w * h
                if(intersection/areamin>.8):
                    mask[toremove]=False

    picked_boxes=[picked_boxes[i] for i in range(len(picked_boxes)) if mask[i]]
    picked_score=[picked_score[i] for i in range(len(picked_score)) if mask[i]]
    output_labels=[output_labels[i] for i in range(len(output_labels)) if mask[i]]
    #print(picked_boxes, picked_score)
    
    return picked_boxes, picked_score, output_labels

colors = {'damage_side_body': (0, 0, 255),
        'damage_front_body': (0, 255, 255),
        'damage_rear_body': (255, 0, 255),
        'damage_side_window': (0, 255, 0),
        'damage_front_window': (255, 140, 0),
        'damage_rear_window': (30, 144, 255),
        'damage_wheel': (255, 255, 0),
        'damage_light': (255, 192, 203),
        'whole_side_body': (30, 144, 255),
        'whole_front_body': (0, 255, 255),
        'whole_rear_body': (255, 0, 0),
        'whole_side_window': (255, 192, 203),
        'whole_front_window': (255, 0, 255),
        'whole_rear_window': (255, 140, 0),
        'whole_wheel': (0, 255, 0),
        'whole_light': (255, 255, 0)}


def save_image_pred(image_name, picked_boxes, picked_score, output_labels):

    # Read image
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.cv2.COLOR_RGB2RGBA)
    overlay = image.copy()
    height, width, channels = image.shape
    
    # Draw parameters
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = .1
    font_scale = min(width,height)/(110/scale)
    alpha = 0.2
    limit = 0.2
   
    # Draw bounding boxes and confidence score after non-maximum supression
    for (start_x, start_y, end_x, end_y), confidence, label in zip(picked_boxes, picked_score, output_labels):
        (w, h), baseline = cv2.getTextSize(f'{label}: {round(confidence*100)}%', font, font_scale, thickness=1)
        cv2.rectangle(img=image, pt1=(start_x, start_y), pt2=(end_x, end_y),
                      color=colors[label], thickness=0)
        
        
        if abs(start_y) < limit*height:
            cv2.rectangle(img=image, pt1=(start_x, start_y + (2 * baseline + 7)), pt2=(start_x + w, start_y - 3),
                          color=colors[label], thickness=-1)
            cv2.putText(image, f'{label}: {round(confidence*100)}%',
                     (start_x, start_y + (2 * baseline + 7)), font, font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.rectangle(img=overlay, 
                          pt1=(start_x, start_y + (2 * baseline + 7)),
                          pt2=(start_x + w, start_y - 3),
                          color=(0, 0, 0), thickness=0)

            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
        else:
            cv2.rectangle(img=image, pt1=(start_x, start_y - (2 * baseline + 7)), pt2=(start_x + w, start_y+3),
                          color=colors[label], thickness=-1) 
            cv2.putText(image, f'{label}: {round(confidence*100)}%',
                     (start_x, start_y), font, font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.rectangle(img=overlay, 
                          pt1=(start_x, start_y - (2 * baseline + 7)),
                          pt2=(start_x + w, start_y+3),
                          color=(0, 0, 0), thickness=0)

            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
         
    cv2.imwrite('predicted.jpg', image_new)
    return


################################## ------ TO STORE PREDICTIONS --------  #############################

try:
    os.mkdir('predictions')
except:
    pass

c = 0
tot = len(os.listdir(img_dir))

start = time.time()
for ws in weights:
    for filename in os.listdir(img_dir):

        os.system(f'./darknet detector test data/obj.data cfg/yolov4-obj.cfg\
                {ws} {img_dir}/{filename} -thresh 0.2 -out results.json\
                -ext_output results.txt')

        coords = get_coords('results.json', filename)
        os.remove('results.json')
        new_coords, scores, labels = non_max_suppression(coords)
        save_image_pred(img_dir + '/' + filename, new_coords, scores, labels) 

        c+=1
        print(f'{c} images over {tot} predicted!')

        shutil.move('predicted.jpg',f'./predictions/{filename.split(".")[0]}_predictions.jpg')
        print(f'{ws} finished!')

shutil.move(src='predictions', dst='../predictions')
end = time.time()


print(f'Saved {len(os.listdir(img_dir))} images with bounding boxes!')
print(f'Prediction time: {round(end - start, 0)} seconds!')


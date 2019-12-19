from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
from imutils import paths
import numpy as np
import argparse
import cv2
import csv
import os

# predicts all frames of a given video
# modified from https://github.com/kapil-varshney/esri_retinanet/blob/master/predict.py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help='path to pre-trained model')
ap.add_argument("-l", "--labels",
	help="path to class labels")
ap.add_argument("-i", "--video", required=True,
	help="path to video file")
ap.add_argument("-o", "--output", default='output',
	help="path to file to store predictions")
ap.add_argument("-c", "--confidence", type=float, default=0.0,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the class label mappings
LABELS = open(args["labels"]).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load the model from disk and grab the video
model = models.load_model(args["model"], backbone_name='resnet50')
video_in = cv2.VideoCapture((args['video']))

# prep output vid
success, img = video_in.read()
height, width, _ = img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(args['output'], fourcc, 30, (width,height))

# loop over the frames
fps = 10
count = 0
while success:

    if count % fps == 0:
        # only predict on certain amount of images
        # set the in between frames to the same as the first frame
        rectangle_input = []
        text_input = []
        print ("[INFO] predicting on frame {}".format(count))

        #load the input image and preprocess it
        image = preprocess_image(img)
        (image, scale) = resize_image(image)
        image = np.expand_dims(image, axis=0)
        # detect objects in the input image and correct for the image scale
        (boxes, scores, labels) = model.predict_on_batch(image)
        boxes /= scale

        # loop over the detections
        for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
            # filter out weak detections
            if score < args["confidence"]:
                continue
                
            print("LABEL", LABELS[label])
            
            # convert the bounding box coordinates from floats to integers
            box = box.astype("int")

            # create the row for each prediction in the format:
            # <xmin> <ymin> <xmax> <ymax> <classname> <confidence>
            box = (box[0], box[1], box[2], box[3], LABELS[label], str(score))
            
            rectangle_input.append(((box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2))
            # adding 50 to y since it's usually around 0 and cuts some off
            text_input.append((LABELS[label] + ' ' + str(score), (box[0], box[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1))
#             print("REC",rectangle_input)
#             print("TEXT", text_input)
    
    # save vid
    if rectangle_input and text_input:
        for r_i, t_i in zip(rectangle_input, text_input):
            img = cv2.rectangle(img, *r_i)
            img = cv2.putText(img, *t_i)
    video_out.write(img)
    success, img = video_in.read()
    count += 1

cv2.destroyAllWindows()
video_out.release()
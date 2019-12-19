from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
from imutils import paths
import numpy as np
import argparse
import csv
import os

# batch predicts on a folder of images
# modified from https://github.com/kapil-varshney/esri_retinanet/blob/master/predict.py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help='path to pre-trained model')
ap.add_argument("-l", "--labels",
	help="path to class labels")
ap.add_argument("-i", "--images", required=True,
	help="path to directory containing input images")
ap.add_argument("-o", "--output", default='output',
	help="path to file to store predictions")
ap.add_argument("-c", "--confidence", type=float, default=0.0,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the class label mappings
LABELS = open(args["labels"]).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load the model from disk and grab all input image paths
model = models.load_model(args["model"], backbone_name='resnet50')
imagePaths = list(paths.list_images(args["images"]))

# loop over the input image paths
output_list = []
for (i, imagePath) in enumerate(imagePaths):

    print ("[INFO] predicting on image {} of {}".format(i+1, len(imagePaths)))

    #load the input image (BGR), clone it, and preprocess it
    image = read_image_bgr(imagePath)
    image = preprocess_image(image)
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

        # convert the bounding box coordinates from floats to integers
        box = box.astype("int")

        # create the row for each prediction in the format:
        # <imagepath> <xmin> <ymin> <xmax> <ymax> <classname> <confidence>
        row = [imagePath, str(box[0]), str(box[1]), str(box[2]), str(box[3]), LABELS[label], str(score)]
        output_list.append(row)

# write the file
with open(args["output"], 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
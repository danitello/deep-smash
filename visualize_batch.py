import argparse
import cv2
import csv
import os
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
        https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
    """
    l.sort(key=alphanum_key)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to directory containing input images")
ap.add_argument("-l", "--boxes",
	help="path to csv of boxes and labels")
ap.add_argument("-o", "--output", default='output',
	help="path to file to store video")
args = vars(ap.parse_args())

# prep boxes
# stored as bndbox[frame_#.jpg] = (xmin, ymin, xmax, ymax, label, confidence)
bndbox = {}
with open(args['boxes']) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        name = os.path.basename(row[0])
        if bndbox.get(name) != None:
            bndbox[name].append((row[1], row[2], row[3], row[4], row[5], row[6]))
            continue
        bndbox[name] = [(row[1], row[2], row[3], row[4], row[5], row[6])]
# prep images
images = [img for img in os.listdir(args['images'])]
sort_nicely(images)

frame = cv2.imread(os.path.join(args['images'], images[0]))
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(args['output'], fourcc, 5, (width,height))

# write
for image_file in images:
    img = cv2.imread(os.path.join(args['images'], image_file))
    print(os.path.join(args['images'], image_file))
    boxes = bndbox.get(image_file)
    if boxes != None:
        for box in boxes:
            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 5)
            # adding 50 to y since it's sometimes around 0 and cuts some off
            img = cv2.putText(img, box[4] + ' ' + box[5], (int(box[0]), int(box[1]) + 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    video.write(img)

cv2.destroyAllWindows()
video.release()
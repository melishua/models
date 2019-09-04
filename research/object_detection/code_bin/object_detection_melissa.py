###############################################################################
#                                                                             #  
#                  Object Detection Source Code For ISee                      #
#                                                                             #  
###############################################################################

#------------------------------------------------------------------------------
# Libraries 
#------------------------------------------------------------------------------
# TensorFlow API
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

# For file check in directory
import os.path
from os import path

# For text to speech
import pyttsx3

#------------------------------------------------------------------------------
# Constant declaration
#------------------------------------------------------------------------------
# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
# Define score needed to accept the output
tf_min_score_thresh = 0.6

# Initiate speech engine
speech_engn = pyttsx3.init()
notify_perct_thresh = 0.3

#------------------------------------------------------------------------------
# Detection Functions
#------------------------------------------------------------------------------

# Function: TensorFlowDetection
# Input: None
# Output: Video from the camera with labelled class and score
# Description: 
#   It is a modified version of the object detection API from Tensorflow. The
#   purpose of this function is to be able to detect basic objects in video
#   stream coming from camera source
def TensorFlowDetection():
    # What model to download.
    # Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 90

    # Download Model if needed
    if not path.exists(MODEL_NAME):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                # Read frame from camera
                ret, image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Print result to console
                # print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > tf_min_score_thresh])

                # Initiate facial recongition if detected a person
                if isTherePerson(classes, scores, category_index):
                    # Call second API for facial recongition
                    print("There is a person!");
                    #getCoordination(boxes)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Display output
                cv2.imshow('object detection', image_np)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

#------------------------------------------------------------------------------
# Voice Notification Functions
#------------------------------------------------------------------------------
def voiceNotification(str_txt):
    speech_engn.say(str_txt)
    speech_engn.runAndWait()

def notifyName(name):
    notification = ""
    if name.lower() == "unkown":
        notification = "Hi there, nice to meet you"
    else:
        notification = "Hi " + name
    voiceNotification(notification)

def notifyObject(obj):
    notification = "Please be careful, there is a " + obj + " in front of you"
    voiceNotification(obj)

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------
# def facialRecongition():
#     print ("Hi")

# def filterOutput():
#     print ("Hi")

def isTherePerson(classes, scores, category_index):
    for index,value in enumerate(classes[0]):
        if scores[0,index] > tf_min_score_thresh:
            # example output: {'id': 1, 'name': 'person'}
            out_dict = category_index.get(value)
            if out_dict['id'] == 1:
                return True
    return False

def getCoordination(boxes):
    ymin = int((boxes[0][0][0]*height))
    xmin = int((boxes[0][0][1]*width))
    ymax = int((boxes[0][0][2]*height))
    xmax = int((boxes[0][0][3]*width))

    result = np.array([ymin,ymax,xmin,xmax])
    print(Result)

if __name__ == "__main__":
    try:
        TensorFlowDetection()
    except Exception as e:
        raise e

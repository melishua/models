###############################################################################
#                                                                             #  
#                  Object Detection Source Code For ISee                      #
#                                                                             #  
#-----------------------------------------------------------------------------#
#                                                                             #  
# Description:                                                                #  
# This is a demo of running face recognition on live video from your webcam.  #
# It's a little more complicated than the other example, but it includes some #
# basic performance tweaks to make things run a lot faster:                   #
#   1. Process each video frame at 1/4 resolution (though still display it at #
#      full resolution)                                                       #
#   2. Only detect faces in every other frame of video.                       #
#                                                                             #
# NOTE: This example requires OpenCV (the `cv2` library) to be installed only #
# to read from your webcam. OpenCV is NOT required to use the face_recognition#
# library. It's only required if you want to run this specific demo. If you   #
# have trouble installing it, try any of the other demos that don't require it#
# instead.                                                                    #
###############################################################################

#------------------------------------------------------------------------------
# Libraries Import
#------------------------------------------------------------------------------
import numpy as np
import cv2
import face_recognition
import pyttsx3

#------------------------------------------------------------------------------
# Constants / Global Declaration
#------------------------------------------------------------------------------
# Flag for showing video stream
cv_show_image_flag = False # Keep false until cv2 crash is resolved
# Flag for outputing audio notification
# *TODO*: 
#   The problem crush currently when both video and audio output is enable!!!!
#   So pyttsx3_output_audio is set to !(cv_show_image_flag) for now. The value
#   can be change to specific value when the bug is fixed.
pyttsx3_output_audio = not cv_show_image_flag

# Default path for adding new annotation file
ANNOTATION_PATH = "./known_annotation"

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
# Camera source - Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Audio source - Initiate speech engine
speech_engn = pyttsx3.init()
notify_perct_thresh = 0.3

#------------------------------------------------------------------------------
# Load Face and annotation records

# Load a sample picture and learn how to recognize it.
angel_image = face_recognition.load_image_file("./known_ppl/Angel Gao.jpg")
angel_face_encoding = face_recognition.face_encodings(angel_image)[0]

# Load a second sample picture and learn how to recognize it.
melissa_image = face_recognition.load_image_file("./known_ppl/Melissa Pan.jpg")
melissa_face_encoding = face_recognition.face_encodings(melissa_image)[0]

# Load a third picture and learn how to recognize it.
stella_image = face_recognition.load_image_file("./known_ppl/Stella Tao.jpg")
stella_face_encoding = face_recognition.face_encodings(stella_image)[0]

# Load a fourth picture and learn how to recognize it.
steve_image = face_recognition.load_image_file("./known_ppl/Steve Mann.jpg")
steve_face_encoding = face_recognition.face_encodings(steve_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    angel_face_encoding,
    melissa_face_encoding,
    stella_face_encoding,
    steve_face_encoding
]

known_face_names = [
    "Angel Gao",
    "Melissa Pan",
    "Stella Tao",
    "Steve Mann"
]

#------------------------------------------------------------------------------
# Face Recongition Function
#------------------------------------------------------------------------------
def face_recognition_webcam():
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color 
        # (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Mainline Recongition
        #------------------------------------------------------------------------------
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    # If a match was found in known_face_encodings, just use the first one.
                    # first_match_index = matches.index(True)
                    # name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]
                        print (name + " " + str(best_match_index))

                face_names.append(name)

                # Audio Notfication
                if pyttsx3_output_audio:
                    notifyNameAndInfo(name, best_match_index)

                # Feature to add new person or annotation


        # Display the results
        if cv_show_image_flag:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Update frame control knob
        process_this_frame = not process_this_frame

#------------------------------------------------------------------------------
# Voice Notification Functions
#------------------------------------------------------------------------------
def voiceNotification(str_txt):
    speech_engn.say(str_txt)
    speech_engn.runAndWait()

def notifyNameAndInfo(name, id):
    notification = ""
    annotations = []

    # Format notification message
    if name == "Unknown":
        notification = "Hi there, nice to meet you"
    else:
        notification = "Hi " + name
        annotations = readAnnotationFromId(id)

    # Notify name
    voiceNotification(notification)

    # Notify annotations
    for a in annotations:
        voiceNotification(a)

#------------------------------------------------------------------------------
# Annotation Related Functions
#------------------------------------------------------------------------------
def readAnnotationFromId(id):
    # Get matching annotation file from database
    af, found = getAnnotationByFRId(id)

    if not found:
        return []

    # Read Annotation File
    f = open(af, "r")

    annotations = f.readlines()

    # close the file after reading the lines.
    f.close()

    return annotations

def annotateById(id, annotation):
    # Get matching annotation file from database
    # af, found = 

    f = open(af, "a+")

    # Create new file if annotation not found
    if not found:
        af = ANNOTATION_PATH + name + ".txt"
        f = open(af, "w+") 

    # Append new annotation line by line
    for a in annotation:
        f.write(a + '\n')

    # close the file after writing the lines.
    f.close()

#------------------------------------------------------------------------------
# Add New Faces Functions
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Database functions
#------------------------------------------------------------------------------

def getAnnotationByFRId(id):
    # hardcode for now 
    return (ANNOTATION_PATH + "/"+ str(id) + ".txt", True)

# def updateAnnotationByFR(name,path):
# def idToName(id):
# def nameToId(name):


#------------------------------------------------------------------------------
# Cleanup Functions
#------------------------------------------------------------------------------
def generalCleanup():
    # Turn off audio source
    speech_engn.stop()
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

#------------------------------------------------------------------------------
# Run Program
#------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        face_recognition_webcam()
        generalCleanup()

    except Exception as e:
        raise e

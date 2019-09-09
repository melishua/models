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
import os
import re
import glob
import speech_recognition as sr
from os import path

#------------------------------------------------------------------------------
# Constants / Global Declaration
#------------------------------------------------------------------------------
# Flag for showing video stream
CV_SHOW_IMAGE_FLAG = False # Keep false until cv2 crash is resolved
# Flag for outputing audio notification
# *TODO*: 
#   The problem crush currently when both video and audio output is enable!!!!
#   So PYTTSX3_OUTPUT_AUDIO is set to !(CV_SHOW_IMAGE_FLAG) for now. The value
#   can be change to specific value when the bug is fixed.
PYTTSX3_OUTPUT_AUDIO = not CV_SHOW_IMAGE_FLAG

# Default path for adding new annotation file
ANNOTATION_PATH = "./known_annotation"
LAST_UNKNOWN_PERSON = "Unknown"
UNKNOWN_PERSON_IDX = 1
STARING_COUNTER = 0
STARING_THRESHOLD = 2
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
# Camera source - Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Audio source - Initiate speech engine
speech_engn = pyttsx3.init()
notify_perct_thresh = 0.3

# speech recognition by obtaining audio from the microphone
speech_recognizer = sr.Recognizer()

#------------------------------------------------------------------------------
# Function to load Face and annotation records
#------------------------------------------------------------------------------
def LoadFaceAndEncoding(known_ppl_pics):
    # Load a picture of each person and learn how to recognize it.
    for known_person_pic in known_ppl_pics:
        # get this person's name
        image_name = os.path.basename(known_person_pic)
        person_name = os.path.splitext(image_name)[0]

        # get this person's face encoding
        image = face_recognition.load_image_file(known_person_pic)
        face_encoding = face_recognition.face_encodings(image)[0]
        
        #TODO: save this person's name and face encoding in DB!
        # save this person's name and face encoding
        KNOWN_FACE_NAMES.append(person_name)
        KNOWN_FACE_ENCODINGS.append(face_encoding)

        print("I can recognize " + person_name + " now.")



#------------------------------------------------------------------------------
# Function to record an unknown person
#------------------------------------------------------------------------------
def RecordUnknownPerson(face_encoding):
    global UNKNOWN_PERSON_IDX
    global LAST_UNKNOWN_PERSON
    global KNOWN_FACE_ENCODINGS
    global KNOWN_FACE_NAMES

    #assign a name to his/her
    person_name = "Unknown_" + str(UNKNOWN_PERSON_IDX)

    #set LAST_UNKNOWN_PERSON to this person
    LAST_UNKNOWN_PERSON = person_name

    #TODO: save encoding in DB!
    #save his/her face encoding
    KNOWN_FACE_NAMES.append(person_name)
    KNOWN_FACE_ENCODINGS.append(face_encoding)

    print("I can recognize " + person_name + " now.")    

    #increment UNKNOWN_PERSON_IDX
    UNKNOWN_PERSON_IDX += 1

def AddUnknownAsContact():
    global LAST_UNKNOWN_PERSON
    global KNOWN_FACE_NAMES
    global STARING_COUNTER

    contact_request = "Would you like to add this person as a contact?"
    answer = getTextFromAudio(contact_request)
    if "yes" in answer:
        name_request = "What is his or her name?"
        name_of_unknown_person = getTextFromAudio(name_request)

        # update person name
        # TODO: update in DB!
        for idx, known_face_name in enumerate(KNOWN_FACE_NAMES):
            if LAST_UNKNOWN_PERSON in known_face_name:
                print("Update " + known_face_name + " to " + name_of_unknown_person)
                KNOWN_FACE_NAMES[idx] = name_of_unknown_person

                # confirm updated contact
                contact_added = getTextFromAudio(name_of_unknown_person + "is added to your contact list.")
                print("Here is the updated contact list:")

        print(KNOWN_FACE_NAMES)

        # restart counting since we already added this person
        # TODO: more handling should go in to this
        STARING_COUNTER = 0

    else:
        # restart counting since we didn't want to add this person
        # TODO: more handling should go in to this
        STARING_COUNTER = 0
        print("No person is added to your contact.")

#------------------------------------------------------------------------------
# Face Recongition Function
#------------------------------------------------------------------------------
def FaceRecognitionWebcam():
    # TODO: check if database is empty
    # if yes, load Face and annotation records from database and save them
    # if no, then do not need to record these people again
        # load pictures of known people from known_ppl_path
    known_ppl_pics = glob.glob("./known_ppl/*.jpg")
    LoadFaceAndEncoding(known_ppl_pics)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    global STARING_COUNTER
    global STARING_THRESHOLD
    global KNOWN_FACE_NAMES
    global KNOWN_FACE_ENCODINGS

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
                matches = face_recognition.compare_faces(KNOWN_FACE_ENCODINGS, face_encoding)
                name = "Unknown"

                if True in matches:
                    # If a match was found in KNOWN_FACE_ENCODINGS, just use the first one.
                    # first_match_index = matches.index(True)
                    # name = KNOWN_FACE_NAMES[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                        name = KNOWN_FACE_NAMES[best_match_index]

                        # count the number of times this unknown person is seen consecutively
                        if name == LAST_UNKNOWN_PERSON:
                            # increment STARING_COUNTER for this particular unknown person
                            STARING_COUNTER += 1
                            print("STARING_COUNTER is: " + str(STARING_COUNTER))

                            # if you continue to stare at this unknown person
                            # i.e. might be having a conversation with
                            # prompt to add this person as a contact
                            if STARING_COUNTER > STARING_THRESHOLD:
                                AddUnknownAsContact()

                    else:
                        #TODO: we should associate staring counter with each unknown person in DB
                        # Since we are now looking at another unknown person, restart counting
                        STARING_COUNTER = 0
                        RecordUnknownPerson(face_encoding)
                        #increment STARING_COUNTER for this particular unknown person
                        STARING_COUNTER += 1
                        print("recognized new unknown person: " + LAST_UNKNOWN_PERSON)
                        print("STARING_COUNTER is: " + str(STARING_COUNTER))

                face_names.append(name)

                # Audio Notfication
                # TODO: set up a timer for voice notification so we don't keep repeat ourselves
                if PYTTSX3_OUTPUT_AUDIO:
                    notifyNameAndInfo(name, best_match_index)

                # Feature to add new person or annotation


        # Display the results
        if CV_SHOW_IMAGE_FLAG:
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
def VoiceNotification(str_txt):
    speech_engn.say(str_txt)
    speech_engn.runAndWait()

def notifyNameAndInfo(name, id):
    notification = ""
    annotations = []

    # Format notification message
    if "Unknown" in name:
        notification = "There is an unknown person in front you."
    else:
        notification = "This is " + name
        annotations = readAnnotationFromId(id)

    # Notify name
    VoiceNotification(notification)

    # Notify annotations
    for a in annotations:
        VoiceNotification(a)

#------------------------------------------------------------------------------
# Speech Recognition Functions
#------------------------------------------------------------------------------
def getTextFromAudio(indicator):
    with sr.Microphone() as source:
        VoiceNotification(indicator)
        audio = speech_recognizer.listen(source)

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        text = speech_recognizer.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))



#------------------------------------------------------------------------------
# Annotation Related Functions
#------------------------------------------------------------------------------
def readAnnotationFromId(id):
    # Get matching annotation file from database
    af, found = getAnnotationByFRId(id)

    if found == False:
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
    personal_annotation = ANNOTATION_PATH + "/"+ str(id) + ".txt"
    if os.path.exists(personal_annotation) == True:
        return (personal_annotation, True)
    else:
        return ('No such file', False)

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
        FaceRecognitionWebcam()
        generalCleanup()

    except Exception as e:
        raise e

###############################################################################
#                                                                             #  
#                  Object Detection Source Code For ISee                      #
#                                                                             #  
#-----------------------------------------------------------------------------#
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
import face_recognition
import cv2
import numpy as np
import pyttsx3
import os
import re
import glob

#------------------------------------------------------------------------------
# Constants Declaration
#------------------------------------------------------------------------------
# Flag for showing video stream
cv_show_image_flag = True # Keep false until cv2 crash is resolved
# Flag for outputing audio notification
# *TODO*: pyttsx3_output_audio is currently !(cv_show_image_flag) due to crash
#         of program when both are enable. The value can be changed to whatever
#         option once the bug is fixed!!!
pyttsx3_output_audio = not cv_show_image_flag
last_unknown_person = "Unknown"

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
# Camera source - Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Audio source - Initiate speech engine
speech_engn = pyttsx3.init()
notify_perct_thresh = 0.3

#------------------------------------------------------------------------------
# Function to load Face and annotation records
#------------------------------------------------------------------------------
known_face_encodings = []
known_face_names = []

def load_face_and_encoding(known_ppl_pics):
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
        known_face_names.append(person_name)
        known_face_encodings.append(face_encoding)

        print("I can recognize " + person_name + " now.")



#------------------------------------------------------------------------------
# Function to record an unknown person
#------------------------------------------------------------------------------
# def record_unknown_person(4_points):
    #snap a photo of this person

    #TODO: save picture in DB!
    #save it locally

    #assign a name to his/her - e.g. Unknown_1
    #set last_unknown_person to this person
    #load_face_and_encoding()

    #TODO: save encoding in DB!
    #save his/her face encoding





#------------------------------------------------------------------------------
# Face Recongition Function
#------------------------------------------------------------------------------
def face_recognition_webcam():
    # TODO: check if database is empty
    # if yes, load Face and annotation records from database and save them
    # if no, then do not need to record these people again
        # load pictures of known people from known_ppl_path
    known_ppl_pics = glob.glob("./known_ppl/*.jpg")
    load_face_and_encoding(known_ppl_pics)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    staring_counter = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

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
                        #TODO:
                        # count the number of times this unknown person is seen consecutively
                        # if name == last_unknown_person:
                            # increment staring_counter for this particular unknown person
                            # if staring_counter > 35:
                                #print("Would you like to add this person as a contact? (y/n)")
                                # take in response
                                #print("Name: ")
                                # take in response
                                # update jpg name, person name (and in DB)

                else:
                    #TODO:
                    # Since we are now looking at another unknown person
                    staring_counter = 0
                    #record_unknown_person(face_locations)
                    #set last_unknown_person to this person
                    #increment staring_counter for this particular unknown person


                face_names.append(name)
                if pyttsx3_output_audio:
                    notifyName(name)

        process_this_frame = not process_this_frame


        # Display the results
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

        if cv_show_image_flag:
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


#------------------------------------------------------------------------------
# Voice Notification Functions
#------------------------------------------------------------------------------
def voiceNotification(str_txt):
    speech_engn.say(str_txt)
    speech_engn.runAndWait()

def notifyName(name):
    notification = ""
    if name == "Unknown":
        notification = "Hi there, nice to meet you"
    else:
        notification = "Hi " + name
    voiceNotification(notification)


if __name__ == "__main__":
    try:
        face_recognition_webcam()
    except Exception as e:
        raise e

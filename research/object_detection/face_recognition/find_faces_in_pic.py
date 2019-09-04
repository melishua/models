from PIL import Image
import face_recognition
import numpy as np

# Load the jpg file into a numpy array
unknown_image = face_recognition.load_image_file("./unknown_pics/unknown.jpg")

angel_image = face_recognition.load_image_file("./known_ppl/Angel Gao.jpg")
angel_face_encoding = face_recognition.face_encodings(angel_image)[0]

melissa_image = face_recognition.load_image_file("./known_ppl/Melissa Pan.jpg")
melissa_face_encoding = face_recognition.face_encodings(melissa_image)[0]

known_face_encodings = [
	angel_face_encoding,
	melissa_face_encoding
]

known_face_names = [
	"Angel Gao",
	"Melissa Pan"
]
	

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations) 

print("I found {} face(s) in this photograph.".format(len(face_locations)))

# for face_location in face_locations:

#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()

#initialize some variables
face_names = []

for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Print the location of each face in this image
    print("{}\'s face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(name, top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = unknown_image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show(title=name)








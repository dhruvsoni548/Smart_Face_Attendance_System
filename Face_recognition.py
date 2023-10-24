import face_recognition 
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

# Load images
devans_image = face_recognition.load_image_file("faces/devans.jpg")
devans_encoding = face_recognition.face_encodings(devans_image)[0]
arihant_image = face_recognition.load_image_file("faces/Arihant.jpg")
arihant_encoding = face_recognition.face_encodings(arihant_image)[0]

known_face_encodings = [devans_encoding, arihant_encoding]
known_face_names = ["Devans", "Arihant"]

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file for writing
with open(f"{current_date}.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    csv_writer.writerow([name, current_time])

                # Draw a rectangle around the recognized face and display the name
                top, right, bottom, left = [i * 4 for i in face_locations[0]]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name + " Present", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

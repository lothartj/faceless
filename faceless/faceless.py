import streamlit as st
import cv2
import numpy as np
import face_recognition

# Function to blur faces in an image using face_recognition library
def blur_faces(image):
    # Find face locations using face_recognition
    face_locations = face_recognition.face_locations(image)

    # Blur each detected face
    for top, right, bottom, left in face_locations:
        face = image[top:bottom, left:right]
        # Convert face to grayscale for blurring
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Apply GaussianBlur to the face in the grayscale image
        face_gray = cv2.GaussianBlur(face_gray, (99, 99), 30)
        # Replace the face in the original image with the blurred face
        image[top:bottom, left:right, 0] = face_gray
        image[top:bottom, left:right, 1] = face_gray
        image[top:bottom, left:right, 2] = face_gray

    return image

# Streamlit app
def main():
    st.title("Face Blur App")

    uploaded_file = st.file_uploader("Choose a video or image file", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            st.image(blur_faces(image), caption="Blurred Faces", use_column_width=True)
        elif file_extension == 'mp4':
            video_file_path = "temp_video.mp4"
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_file_path)

            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(blur_faces(frame))

            cap.release()

            # Combine frames to create a new video
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter("temp_output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

            st.video("temp_output_video.mp4")

if __name__ == "__main__":
    main()

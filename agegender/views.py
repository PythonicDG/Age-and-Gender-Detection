import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from collections import deque
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.http import StreamingHttpResponse
from .forms import ImageUploadForm
from .models import UploadedImage

# Load models
age_model = load_model(r"C:\Users\Dipak\Desktop\Final Year Project\Models\pretrained-age-detection.h5", custom_objects={"mae": MeanAbsoluteError()})
gender_model = load_model(r"C:\Users\Dipak\Desktop\Final Year Project\Models\best_model.keras")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Home Page
def home(request):
    return render(request, "home.html")

# Image Upload & Processing
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            return redirect('uploaded_results', image_id=image_instance.id)

    else:
        form = ImageUploadForm()

    return render(request, "upload_image.html", {"form": form})

# Results Page for Uploaded Image
def uploaded_results(request, image_id):
    image_instance = get_object_or_404(UploadedImage, id=image_id)
    image_path = default_storage.path(image_instance.image.name)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (128, 128)).reshape(1, 128, 128, 1) / 255.0

        predicted_age = int(round(age_model.predict(face_resized)[0][0]))
        gender_prediction = gender_model.predict(face_resized)
        gender_label = "Female" if gender_prediction[0][0] > 0.5 else "Male"

        results.append(f"Age: {predicted_age}, Gender: {gender_label}")

    return render(request, "uploaded_results.html", {"image": image_instance, "results": results})

# Video Streaming Generator for Live Detection
def generate_frames():
    cap = cv2.VideoCapture(0)
    age_predictions = deque(maxlen=10)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (128, 128)).reshape(1, 128, 128, 1) / 255.0

            predicted_age = int(round(age_model.predict(face_resized)[0][0]))
            age_predictions.append(predicted_age)
            smoothed_age = int(sum(age_predictions) / len(age_predictions))

            gender_prediction = gender_model.predict(face_resized)
            gender_label = "Female" if gender_prediction[0][0] > 0.5 else "Male"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {smoothed_age}, {gender_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Live Detection Page
def live_detection(request):
    return render(request, "live_detection.html")

# Live Video Streaming Route
def live_video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    
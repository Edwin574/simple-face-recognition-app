# myapp/views.py

import os
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def upload(request):
    if request.method == 'POST' and request.FILES['image']:
        file = request.FILES['image']
        image_path = default_storage.save('images/' + file.name, file)
        return JsonResponse({"result": "Image uploaded successfully."})
    return render(request, 'index.html')

def train(request):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image_dir = os.path.join(settings.MEDIA_ROOT, 'images')
    faces, ids = get_images_and_labels(image_dir)
    recognizer.train(faces, np.array(ids))
    model_path = os.path.join(settings.STATIC_ROOT, 'dist/models/face_recognition_model.yml')
    recognizer.save(model_path)
    return JsonResponse({"result": "Model trained and saved successfully."})

@csrf_exempt
def recognize(request):
    if request.method == 'POST' and request.FILES['image']:
        file = request.FILES['image']
        image_path = default_storage.save('temp/' + file.name, file)
        result = recognize_image(image_path)
        os.remove(image_path)
        return JsonResponse({"result": result})
    return render(request, 'recognize.html')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    faces = []
    ids = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            id = int(filename.split('_')[0])
            faces.append(image)
            ids.append(id)
        except ValueError:
            print(f"Filename {filename} does not contain a valid ID.")
            continue
    return faces, ids

def recognize_image(image_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = os.path.join(settings.STATIC_ROOT, 'dist/models/face_recognition_model.yml')
    if not os.path.exists(model_path):
        return "Model file not found"
    recognizer.read(model_path)
    face_cascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, "haarcascade_frontalface_default.xml"))
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    result = "No face detected"
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 80:
            result = f"Recognized: ID {id}, Confidence {conf}"
        else:
            result = "Unknown"
    return result

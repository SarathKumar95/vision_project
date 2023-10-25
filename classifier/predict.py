from PIL import Image
from io import BytesIO
import base64
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import PredictedImage
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle
from face_recognition import load_image_file, face_locations, face_encodings
from django.conf import settings


def predict_image(image):
    # Convert the Django uploaded image to a PIL Image
    pil_image = Image.open(image)

    # Load the trained KNN model
    model_path = os.path.join(settings.BASE_DIR, 'trained_knn_model.clf')
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Make predictions on the image
    predictions = predict(pil_image, knn_clf)

    # Process the predictions
    processed_predictions = []
    for name, (top, right, bottom, left) in predictions:
        # You can further process the name and coordinates here if needed
        # For example, you can convert the name to UTF-8
        name = name.encode("UTF-8")
        processed_predictions.append((name, (top, right, bottom, left)))

    # Save the uploaded image and predictions to your database
    pil_image_bytes = BytesIO()
    pil_image.save(pil_image_bytes, format="JPEG")
    image_file = SimpleUploadedFile("uploaded_image.jpg", pil_image_bytes.getvalue())
    
    predicted_image = PredictedImage(image=image_file, predictions=processed_predictions)
    predicted_image.save()

    return processed_predictions

def predict(pil_image, knn_clf, distance_threshold=0.6):
    X_img_path = pil_image  # In this case, you already have the image as a PIL Image
    X_face_locations = face_locations(X_img_path)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_encodings(X_img_path, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

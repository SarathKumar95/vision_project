from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
import os
from face_recognition import load_image_file, face_locations, face_encodings
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Function for face recognition
def predict(X_img_path, knn_clf, distance_threshold=0.6):
    X_img = load_image_file(X_img_path)
    X_face_locations = face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

# View for face recognition API
class FaceRecognitionView(View):
    @csrf_exempt
    def post(self, request, *args, **kwargs):
        # Load the trained KNN model
        with open("trained_knn_model.clf", 'rb') as f:
            knn_clf = pickle.load(f)

        # Retrieve the image file from the request
        image_file = request.FILES.get('image')

        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        # Save the uploaded image to a temporary location
        temp_image_path = os.path.join("temp_dir", image_file.name)
        with open(temp_image_path, 'wb') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # Make predictions on the uploaded image
        predictions = predict(temp_image_path, knn_clf)

        # Clean up the temporary image file
        os.remove(temp_image_path)

        result = {'predictions': []}
        for name, _ in predictions:
            result['predictions'].append({'name': name})

        return JsonResponse(result, status=200)

# View for rendering the test form
class TestFormView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'test_form.html')

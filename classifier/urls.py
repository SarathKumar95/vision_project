from django.urls import path
from .views import FaceRecognitionView, TestFormView

urlpatterns = [
    path('recognize-face/', FaceRecognitionView.as_view(), name='recognize-face'),
    path('test-form/', TestFormView.as_view(), name='test-form'),
]

# vision_project/urls.py

from django.urls import path
from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='upload.html'), name='classifier'),
    # Add other URL patterns here as needed
]

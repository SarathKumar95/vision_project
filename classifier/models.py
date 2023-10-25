from django.db import models

class PredictedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    predictions = models.JSONField()  # Store predictions as JSON data
    created_at = models.DateTimeField(auto_now_add=True)

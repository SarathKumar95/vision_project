from django.db import models

class RecognizedFace(models.Model):
    label = models.CharField(max_length=255)
    probability = models.FloatField()
    image = models.ImageField(upload_to='uploads/')

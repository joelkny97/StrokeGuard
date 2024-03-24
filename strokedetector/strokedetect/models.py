from django.db import models
from django.utils import timezone
# Create your models here.

class VideoFrames(models.Model):

    frame = models.ImageField(upload_to="video_frames", null=True, blank=True)
    created_timestamp = models.DateTimeField(default=timezone.now)
    stroke_detected = models.BooleanField(default=False)
    prediction_score = models.FloatField(default=0.0)

    def __str__(self) -> str:
        return str(self.id)
    
        




from django.urls import path, include
from . import views
from django.http import StreamingHttpResponse
from .util.camera import generate_frames, VideoCamera

urlpatterns = [
    path('', views.index, name='home'),
    
    path('about/', views.about, name='about' ),
    path('stream/', lambda r: StreamingHttpResponse(generate_frames(VideoCamera()
                                                                    ,content_type='multipart/x-mixed-replace; boundary=frame') )  ),
    path('retrain/', views.retrain, name='retrain'),
]

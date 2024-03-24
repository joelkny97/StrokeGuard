from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('about/', views.about, name='about' ),
    path('retrain/', views.retrain, name='retrain'),
    path('stream/', views.stream_page, name='stream_page'),
    path('streaming/', views.streaming, name='streaming'),
    path('about/', views.about, name='about' ),
]

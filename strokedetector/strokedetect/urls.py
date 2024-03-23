from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('about/', views.about, name='about' ),
    path('detector/', views.detector, name='detector'),
    path('retrain/', views.retrain, name='retrain'),
]

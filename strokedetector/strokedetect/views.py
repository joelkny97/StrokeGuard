from django.shortcuts import render

# Create your views here.



def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def detector(request):
    return render(request, 'detector.html')

def retrain(request):
    return render(request, 'retrain.html')



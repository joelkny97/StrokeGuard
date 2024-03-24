from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseRedirect
from .util.camera import generate_frames, VideoCamera
# from .classifier.stroke_classifier import StrokeDetectorClassifier
import random
from .forms import ImageUploadForm
import base64

# Create your views here.



def index(request):
    context={}
    if request.method == 'POST':
        stroke = False
        # result = StrokeDetectorClassifier().predict()
        result = random.randint(0, 1)
        if result == True:
            stroke = True
        context['stroke'] = stroke
        return render(request, HttpResponseRedirect('index.html'), context=context)
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def stream_page(request):

    return render(request, 'stream.html')
def streaming(request):

    return StreamingHttpResponse(generate_frames(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
    
    # return render(request, 'stream.html', {'stream': stream})



def retrain(request):
    image_uri = None
    predicted_label = None
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # passing the image as base64 string to avoid storing it to DB or filesystem
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label
            try:
                predicted_label = False
            except RuntimeError as re:
                print(re)
                # predicted_label = "Prediction Error"

    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }

    return render(request, 'retrain.html', context)



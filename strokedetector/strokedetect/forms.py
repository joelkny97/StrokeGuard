from django import forms

class ImageUploadForm(forms.Form):
    image = forms.FileField(
        widget=forms.FileInput(attrs={'input': 'file', 'accept': 'image/*', 'capture': 'camera'}),
    )

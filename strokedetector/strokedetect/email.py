from django.core.mail import send_mail

def sendEmail(self, isStroke):
    if (isStroke == True):
        
        send_mail('Stroke', 'This person is having a stroke please help them!' , 'bilalsiddiqi629@gmail.com', 'hprofessional510@gmail.com')

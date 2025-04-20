# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
import os


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def Training(request):
    return render(request, 'users/Ramam_ANP.html', {})


def UserClassification(request):
    return render(request, 'users/cl_reports.html', {})


def UserPredictions(request):
    if request.method == 'POST':
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/plate_test/")
        filename = fs.save(image_file.name, image_file)
        # detect_filename = fs.save(image_file.name, image_file)
        uploaded_file_url = "/media/plate_test/" + filename  # fs.url(filename)
        from .utility.predections import predict_user_input
        result, prob = predict_user_input(filename)
        path = fs.url(os.path.join(settings.MEDIA_ROOT, "plots", "plotted_image.png"))
        return render(request, "users/UploadForm.html", {'path': path, 'result': result})
    else:
        return render(request, "users/UploadForm.html", {})


def UseLiveNow(request):
    from .utility.predections import video_test
    video_test()
    return render(request, 'users/UserHomePage.html', {})


def PlateDetection(request):
    import time
    import cv2
    import numpy as np
    frame_width = 640
    frame_height = 480
    cascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
    min_area = 500
    counter = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    cap.set(10, 150)  ##Changing brightness to 150
    start_time = time.time()
    capture_duration = 30
    while int(time.time() - start_time) < capture_duration:
        success, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        number_plates = cascade.detectMultiScale(img_gray, 1.1, 5)
        for (x, y, w, h) in number_plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(img, "Number Plate", (x, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                """Picking the number plate from the caption"""
                img_roi = img[y:y + h, x:x + w]
                cv2.imshow("Region of Interest", img_roi)

        cv2.imshow("Capture", img)
        """Saving the captured number plate to the Scanned file by pressing the 's' key"""
        if cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite("Resources/Scanned/NoPlate_" + str(counter) + ".jpg", img_roi)
            """Creating a message that will show on the screen when a number plate successfully saved"""
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'users/UserHomePage.html', {})

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
from . import red_emo_hair
import dlib
from math import hypot
import cv2

# Create your views here.

out_dict = {'Heart': 0, 'Oblong': 1, 'Oval': 2, 'Round': 3, 'Square': 4}

li = list(out_dict.keys())

def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        m = str(filename)
        K.clear_session()
        im = Image.open("{}/".format(settings.MEDIA_ROOT) + m)
        j = im.resize((256, 256),)
        l = "predicted.jpg"
        j.save("{}/".format(settings.MEDIA_ROOT) + l)
        file_url = fs.url(l)
        mod = load_model('model1.hdf5', compile=False)
        
        img1 = cv2.imread('C:\\hairRecommendation\\media\\' + filename)

        detector = dlib.get_frontal_face_detector()
        face = detector(img1)

        img = image.load_img(myfile, target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = mod.predict(x)
        d = preds.flatten()
        j = d.max()

        #For the template fitting part, we should import and export the image with the cv2 methods
        processed_image = red_emo_hair.process_red_hair(img1)
        cv2.imwrite("C:\\hairRecommendation\\media\\processed.jpg", processed_image)
        
        # processed_image = Image.fromarray(processed_image)
        # processed_image.save("C:\\Users\\Laravel\\sailoonai\\manish\\djangoMedia\\media\\processed.jpg")
        new_url = fs.url('processed.jpg')

        for index, item in enumerate(d):
            if item == j:
                result = li[index]
                return render(request, "index.html", {
                                'result': result, 'file_url': file_url, 'new_url': new_url, 'face': face })
    return render(request, "index.html")

def aboutus(request):
    return render(request, 'aboutus.html')

    
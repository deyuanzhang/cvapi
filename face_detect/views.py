# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import numpy as np
import urllib
import json
import cv2
import os

# Create your views here.

face_detector = "haarcascade_frontalface_default.xml"

@csrf_exempt
def requested_url(request):

	#no detection, default should be false
	default = {"safety_executed": False}

	if request.method == "POST":	
		if request.FILES.get("image", None) is not None:
			image_to_read = read_image(stream = request.FILES["image"])
		else:
			#url was provided by the user
			url_provided = request.POST.get("url", None)
			if url_provided is None:
				default["error_value"] = "There is no URL provided"
				return JsonResponse(default)
			image_to_read = read_image(url = url_provided)

		image_to_read = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY)

		detector_value = cv2.CascadeClassifier(face_detector)

		values = detector_value.detectMultiScale(image_to_read, scaleFactor=1.1, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

		values = [(int(a), int(b), int(a+c), int(b+d)) for (a,b,c,d) in values]
		default.update({"# of faces": len(values), "faces":values, "safely_executed": True })

	return JsonResponse(default)

def read_image(path=None, stream=None, url=None):
	if path is not None:
		image = cv2.imread(path)
	else:
		if url is not None:
			response = urllib.request.urlopen(url)
			data_temp = response.read()
		elif stream is not None:
			data_temp = stream.read()

		image = np.asarray(bytearray(data_temp), dtype="uint8")

		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	return image
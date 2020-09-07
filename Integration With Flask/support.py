import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import time
import os
import imutils



def detect_and_predict(frame, faceNet, maskNet):    
	#print(frame.shape)
	(h, w) = frame.shape[:2]
	
	#blob = cv2.dnn.bolbFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)
	
	faces = []
	locs = []
	preds = []
	
	
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
		
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
			(startX, startY, endX, endY) = box.astype('int')
			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w-1, endX), min(h-1, endY))
			
			face =  frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	
	if len(faces)>0:
		faces = np.array(faces)
		preds = maskNet.predict(faces, batch_size=32)
		
		
	return (locs, preds)
	
	


prototxtPath = '/home/vishwa/Documents/COVID Project/Technocolabs-Internship-Project/FacemaskDetection/deploy.prototxt'
weightsPath = '/home/vishwa/Documents/COVID Project/Technocolabs-Internship-Project/FacemaskDetection/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


unmasked_dir = '/home/vishwa/Documents/COVID Project/Technocolabs-Internship-Project/FacemaskDetection/Unmasked_Faces'
maskNet = load_model('facefeatures.h5')
	
 
 
 
class util:
	def findandret(frame):
		i = 0
		(locs, preds) = detect_and_predict(frame, faceNet, maskNet)
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(withmask, withoutmask) = pred
		
			label = 'Mask' if withmask > withoutmask else 'No Mask'
			
			
			if label == 'No Mask':
				print('hi')
				cv2.imwrite(os.path.join(unmasked_dir, str(i)+'.jpg'), frame[startY:endY, startX:endX])
				i = i + 1

			color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
		
			label = '{}: {:.2f}%'.format(label, max(withmask, withoutmask)*100)
	   
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)  
	
		return frame

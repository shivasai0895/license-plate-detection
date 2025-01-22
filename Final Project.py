# Import dependencies
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import cv2 # This is the OpenCV Python library
import pytesseract # This is the TesseractOCR Python library
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#import pytesseract # This is the TesseractOCR Python library
# Set Tesseract CMD path to the location of tesseract.exe file
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def carplate_detect(image):
    carplate_haar_cascade = cv2.CascadeClassifier(r'C:\Users\sbandl\Downloads\haarcascade_russian_plate_number.xml')
    carplate_overlay = image.copy() 
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1, minNeighbors=3)

    for x,y,w,h in carplate_rects: 
        cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (0,0,255), 5) 
            
        return carplate_overlay
# Create function to retrieve only the car plate region itself
def carplate_extract(image):
    carplate_haar_cascade = cv2.CascadeClassifier(r'C:\Users\sbandl\Downloads\haarcascade_russian_plate_number.xml')
    carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects: 
            carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extract specific region of interest i.e. car license plate
            
    return carplate_img
# Enlarge image for further processing later on
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image

carplate_img_rgb = cv2.imread(r'C:\Users\sbandl\Downloads\car_image.jpg')
#carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)


# Import Haar Cascade XML file for Russian car plate numbers

# Setup function to detect car plate
cv2.imshow('car',carplate_img_rgb)
detected_carplate_img = carplate_detect(carplate_img_rgb)
cv2.imshow('car',detected_carplate_img)
# Display extracted car license plate image
carplate_extract_img = carplate_extract(carplate_img_rgb)
cv2.imshow('car',carplate_extract_img)
carplate_extract_img = enlarge_img(carplate_extract_img, 150)
cv2.imshow('car',carplate_extract_img)

if cv2.waitKey(0) & 0xff == 27: 
	cv2.destroyAllWindows() 

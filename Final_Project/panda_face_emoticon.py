# Link to gitlab repository: https://git.cs.dal.ca/hanwenz/csci_4261_hanwen_zhang/-/tree/master/Final_Project
# You can clone the final project via: https://git.cs.dal.ca/hanwenz/csci_4261_hanwen_zhang.git
# The idea of this code are come from: https://www.cnblogs.com/warcraft/p/10274889.html, https://blog.csdn.net/QuantumEntanglement/article/details/81491031
# Please read ReadMe.txt carefully before running this code

import sys
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import matplotlib as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def rotate_bound(img, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    ''' Grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine '''
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # Perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH))

def rotate_crop_result(img):
    flag = False
    for angle in range(0,360,30):
        
        img_gray = cv2.cvtColor(rotate_bound(img, angle), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor = 1.2,
            minNeighbors = 4,
            flags = cv2.CASCADE_SCALE_IMAGE,
            minSize = (140,140)
        )
        
        for (x,y,h,w) in faces:
            #img = cv2.rectangle(rotate_bound(img,angle), (x,y), (x+w,y+h), (251,244,243), 0)
            img = rotate_bound(img,angle)[y+1:y+w+5, x-5:x+h+10]
            
        if type(faces).__module__ == "numpy":
                flag = True
                break

    return img

def resize(img):
    # Get the width and height of the image
    x, y = img.shape[0:2] 
    # Scale the image to one fifth of the original size
    resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    # Return the resized image
    return resized

if __name__ == '__main__':
 
    # If your enviroment can use the face_recognition package
    # Use the following code to read the image
    img = cv2.imread(sys.argv[1])  
    
    # If you running on the Mac OS google colab use the following code
    # Use the following code to read the image
    #img = cv2.imread('test1.png')   
    
    # Resize the image
    img = resize(img)
    #Rotation face detection result
    rotated_img = rotate_crop_result(img)
    # Write the rotated image to the folder
    cv2.imwrite("rotated_crop_img.png", rotated_img)
    # Load the rotated image
    image = face_recognition.load_image_file("rotated_crop_img.png")
    # Find all facial features for all faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image, model='large')
    
    face_line = list()#Used to record the contours of the face
    face_point = list()#Used to record the range of images to be intercepted
    face = list()#Used to record faces

    for face_landmarks in face_landmarks_list:

        # Find the outline of the face to use to generate a mask
        chin = face_landmarks['chin']
        left_eyebrow = face_landmarks['left_eyebrow']
        right_eyebrow = face_landmarks['right_eyebrow']
        nose_bridge = face_landmarks['nose_bridge']
        chin.reverse()
        list_all = left_eyebrow + right_eyebrow + chin + [left_eyebrow[0]]
        face_line.append(list_all)
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        d.line(list_all, width=5)
        
    # Create a new full black image of the same size as the original image. Use ImageRAW to draw the outline of the face on it as the mask template  
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    q = ImageDraw.Draw(mask)
    for i in face_line:
        q.line(i, width=5, fill=(255, 255, 255))
    mask.save(r"mask.png")#The picture is written out to be processed by OpenCV
    
    # Generate the mask
    mask = cv2.imread('mask.png')
    h, w = mask.shape[:2]  # Read the width and height of the image
    mask_flood = np.zeros([h + 2, w + 2], np.uint8)*255  # Generate a new image matrix, +2 is the official function requirement
    cv2.floodFill(mask, mask_flood, (0, 0), (255, 255, 255))#Here I use OpenCV's flooded fill to paint the outline white on the outside and black on the inside  

    #A 5*5 convolution check mask is used for closing operation to remove noise
    #I tried the Erosion iteration number of 2 times, which makes a better result  
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    mask = dilation
    
    #Re-read the original image, box out RIO, and hand it to OpenCV
    image = rotated_img
    image[mask == 255] = 255
    cv2.imwrite("face.png",image)
    
    image = cv2.imread("face.png")
    
    #Turn the processed image into a grayscale image
    GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #binary processing
    ret, image = cv2.threshold(GrayImage, 90, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binary_face.png", image)#The picture is written out to be processed by Image resize
    
    
    box = (160, 145, 465, 395)#The part of the background to be replaced
    base_img = Image.open('background.png')
    image = Image.open('binary_face.png')
    image = image.resize((box[2] - box[0], box[3] - box[1]))#Scaling the emoji must be the same size as the place where the background was replaced  
    base_img.paste(image, box)
    base_img.show()
    base_img.save('output_emoji.png')#The output image
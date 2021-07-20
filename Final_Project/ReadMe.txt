If you are going to run this code on Windows system
You need to install the environment and packages following the follow tutorial on YouTube so that you can use the face_recognition packages
The url or the tutorial is: https://www.youtube.com/watch?v=xaDJ5xnc8dc

If you are not going to run this code on Windows system
I highly recommend you to run this code on google colab
If you are going to run this code on google colab
You need to select the "Runtime" on the top bar and then "change runtime type" to GPU
Then you need to run the command "!pip install face_recognition" before running the code
In the code, in the main function, replace the code "img = cv2.imread(sys.argv[1])" by "img = cv2.imread('image1.png')" to read the input image

image1.png, image2.png, image3.png, image4.png are the images used to test the code
After running the code, the code will show and write the output_emoji.png to the folder which is the generated panda face emoticon.
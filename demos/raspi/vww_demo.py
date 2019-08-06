import numpy as np
import tensorflow as tf
import cv2
import PIL 

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../../model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

#Start webcam
camera = cv2.VideoCapture(0)

#Throw away first 30 frames (still warming up camera)
for i in range(30):
	temp, t1 = camera.read()

#Continuously perform inference
while True:

    _ , image = camera.read()
    image = PIL.Image.fromarray(image)

    #Resize images to input dimension shape
    image = image.resize((238, 208))
    input_data = np.array(image)
    input_data = np.expand_dims(input_data, axis=0)

    #Call the interpreter to perform classification
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data[0][1] > output_data[0][0]:
        print('Person detected!')
    else:
        print('No person detected.')

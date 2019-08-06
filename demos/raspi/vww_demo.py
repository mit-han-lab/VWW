import numpy as np
import tensorflow as tf
import subprocess
import PIL 

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../../model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
while True:
    subprocess.call('fswebcam --no-banner -r 640x480 -q vww_image.jpg', shell=True)
    image = PIL.Image.open("vww_image.jpg")
    image.thumbnail((278, 208))
    image = image.crop((0, 0, 238, 208))
    input_data = np.array(image)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data[0][1] > output_data[0][0]:
        print('Person detected!')
    else:
        print('No person detected.')

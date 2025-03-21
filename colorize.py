import cv2
import numpy as np
import os





















































































































































































# Initialize the neural network
def initialize_network():
    prototxt_path = 'models/colorization_deploy_v2.prototxt'
    model_path = 'models/colorization_release_v2.caffemodel'
    kernel_path = 'models/pts_in_hull.npy'
    
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    return net

# Initialize network globally
net = initialize_network()

def colorize_image(image_path):
    bw_image = cv2.imread(image_path)
    if bw_image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Normalize the image
    normalized = bw_image.astype('float32') / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    # Resize the image to the network input size
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict the a and b channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted a and b channels to the original image size
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))

    # Combine with the L channel
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    # Save the colorized image
    base_filename = os.path.basename(image_path)
    colorized_image_path = os.path.join('results', 'colorized_' + base_filename)
    cv2.imwrite(colorized_image_path, colorized)
    
    return image_path, colorized_image_path

def convert_to_black_and_white(image_path):
    color_image = cv2.imread(image_path)
    if color_image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Convert to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    base_filename = os.path.basename(image_path)
    gray_image_path = os.path.join('results', 'bw_' + base_filename)
    cv2.imwrite(gray_image_path, gray_image)

    return gray_image_path

def is_black_and_white(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    # Check if the image is grayscale by comparing all channels
    if len(image.shape) == 2:
        return True
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(image)
        return np.array_equal(b, g) and np.array_equal(b, r)
    
    return False

from sqlalchemy.sql.expression import label
import cv2
import os
import numpy as np
from keras.preprocessing import image



from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# camera = cv2.VideoCapture(0)

IMAGE_SIZE = [224, 224]

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, IMAGE_SIZE)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def model_predict(img_path, model, plant, delete):


    labels = [ 'Tomato__Spider_mites',
            'Tomato__Septoria_leaf_spot',
            'Tomato__Late_blight',
            'Pepper__Bacterial_spot',
            'Tomato__Early_blight',
            'Potato__Late_blight',
            'Tomato__YellowLeaf_Curl_Virus',
            'Tomato__healthy',
            'Potato__healthy',
            'Tomato__Tomato_mosaic_virus',
            'Pepper__healthy',
            'Tomato__Target_Spot',
            'Potato__early_blight',
            'Tomato__Bacterial_spot',
            'Tomato__Leaf_Mold',]


    print('\n\n', img_path, '\n')
    print('\n', labels, '\n\n')
    img = image.load_img(img_path, target_size=IMAGE_SIZE)

    # Preprocessing the image
    # x = image.img_to_array(img)
    x = convert_image_to_array(img_path)
    x = np.array(x).astype('float32')/255
    x = np.reshape(x, IMAGE_SIZE + [3])
    x = np.expand_dims(x, axis=0)

    result = model.predict(x)
    print('\n\nDelete:\t', delete, '\n\n')
    if delete:
        os.remove(img_path)
        print('deleted ', img_path)
    print()

    what_class = np.argmax(result, axis=-1)
    scale = '{:.2f}'.format(round(result.max(), 2))

    result_label = labels[what_class[0]]
    plant = result_label.split('__')[0]
    category = result_label.split('__')[-1].replace('_', ' ')

    if result_label.lower() == 'healthy':
        return (f'{plant.title()} leaf is classified HEALTHY with scale of {scale}'), plant.lower(), "Healthy", scale

    else:
        return (f'{plant.title()} leaf is infected with "{category.upper()}" with confidence scale of {scale}'), plant.lower(), category.upper(), scale

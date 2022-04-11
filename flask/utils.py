
import base64
from PIL import Image
import numpy as np
from io import BytesIO
from PIL import Image

def resize_img(img):
    resized_img = img.resize((400, 400))
    return resized_img

def base64_to_img(text):
    _type, data = text.split(',')
    _type = _type.split('/')[1].split(';')[0]

    decoded = base64.b64decode(data)
    img = np.asarray(Image.open(BytesIO(decoded)))
    arr = np.transpose(img, (2, 0, 1))
    return arr, _type

def img_to_base64(arr, _type):
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)).astype('uint8'))
    img = resize_img(img)

    rawBytes = BytesIO()

    img.save(rawBytes, _type)
    rawBytes.seek(0)
    b64_string = base64.b64encode(rawBytes.read())

    result = "data:image/" + _type + ";base64," + str(b64_string, 'utf-8')
    return result
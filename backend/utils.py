import cv2
import numpy as np

def preprocess_image(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    else:
        img = np.zeros((28, 28, 1), dtype=np.float32)
        return img.reshape(1, 28, 28, 1)

    
    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20 / w)))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
    img = canvas

    
    img = np.rot90(img, k=2)  

   
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img
import cv2
import os
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'teseract/tesseract.exe'


def get_digit(img, folder, name):
    cv2.imwrite(folder + '/' + name + '.jpg', img)
    result = pytesseract.image_to_string(image=Image.fromarray(img),
                                         config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')
    return result

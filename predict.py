import numpy as np
from PIL import Image

from facenet import Facenet as mynet

if __name__ == "__main__":
    model = mynet()
    while True:
        img1 = input('Input the image1 path:')
        try:
            img1 = Image.open(img1)
        except:
            print('Wrong input! Please input the right path!')
            continue

        img2 = input('Input the image2 path:')
        try:
            img2 = Image.open(img2)
        except:
            print('Wrong input! Please input the right path!')
            continue
        mark = model.detect_image(img1, img2)
        thredhold = 1.14000
        print(mark)
        print(mark < thredhold)

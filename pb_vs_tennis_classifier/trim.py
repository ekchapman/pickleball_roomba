# Script to remove small thumbnail images from the output of an image crawler
# https://github.com/YoongiKim/AutoCrawler

import os
from PIL import Image

for root, dirs, files in os.walk('C:/Users/echapman/Downloads/AutoCrawler/download/'):
    for filename in files:
        path = os.path.join(root, filename)
        with Image.open(path) as im:
            size = im.size
        
        if size[0] < 20 and size[1] < 20:
            os.remove(path)
            print('removed file of size: {}'.format(im.size))
    for dirname in dirs:
        print(os.path.join(root, dirname))
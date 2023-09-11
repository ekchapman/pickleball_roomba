import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

sizes = []

for root, dirs, files in os.walk('C:/Users/echapman/Downloads/AutoCrawler/download/'):
    for filename in files:
        path = os.path.join(root, filename)
        with Image.open(path) as im:
            sizes.append(im.size)
sizes = np.array(sizes)
print(sizes.shape)

plt.hist(sizes[:, 0])
plt.title('width')

plt.figure()
plt.hist(sizes[:, 1])
plt.title('height')

plt.show()

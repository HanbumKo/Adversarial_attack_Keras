import glob
import numpy
from PIL import Image
import numpy as np
import csv

path = "images/"

files = [f for f in glob.glob(path + "*.png")]
images = []
labels = []

count = 0
for f in files:
    imageName = f[7:-4]
    with open('images.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if row[0] == imageName:
                labels.append(row[6])
                break
    newNp = np.asarray(Image.open(f))
    newList = newNp.tolist()
    images.append(newList)
    print(count, f)
    count += 1

images = np.asarray(images)
labels = np.asarray(labels)
np.save("imageNumpy", images)
np.save("labels", labels)
print(images.shape)
print(labels.shape)


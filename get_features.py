import cv2
import numpy as np
import pandas as pd

with open('./digits/files.txt', 'r') as file:
    files_names = [line.split() for line in file]

moments = []

for f in files_names:
    # print('reading: ' + f[0] + ' - ' + f[1])
    image = cv2.imread('digits/' + f[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hu_moments = (cv2.HuMoments(cv2.moments(image)).flatten())
    moments.append(np.append(f[1], hu_moments))

features = np.array(moments)   
df = pd.DataFrame(features, columns = ['label', 'h1','h2','h3', 'h4', 'h5', 'h6', 'h7'])
df.to_csv('digits/features.csv', index=False)
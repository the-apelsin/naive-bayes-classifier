#DISCLAIMER: Make sure zip.train and zip.test are created and are IMAGES. CIFAR-10 Works Best!

import os
import math

#==== configuration ====#
# main_dir must be the directory containing zip.train and zip.test.
main_dir = 'c:\\users\\your-name-here\\desktop' # REPLACE "your-name-here" with username if zip.train and zip.test are on desktop, otherwise replace with directory.
#=======================#

def load_images(fname):
    images = [[] for i in range(10)]

    os.chdir(main_dir)
    with open(fname, 'r') as f:
        for line in f:
            values = line.split()
            images[int(float(values[0]))].append([float(x) for x in values[1:]])
    return images

mu = []
sigma = []
py = []

##########
# Training
##########

images = load_images('zip.train')
for i in range(10):             # for each digit
    mu.append([])
    sigma.append([])
    for j in range(256):        # for each pixel
        values = [x[j] for x in images[i]]
        tmp_mu = sum(values) / len(values)
        tmp_sigma = math.sqrt(sum([x*x for x in values]) / len(values) - tmp_mu**2)
        mu[i].append(tmp_mu)
        sigma[i].append(0.5)        # 0.5 is better than tmp_sigma!

# Compute P(Y=k) = (# images of digit k) / (# images).
for i in range(10):
    py.append(len(images[i]))
num_images = sum(py)
py = [float(x) / num_images for x in py]

#########
# Testing
#########

images = load_images('zip.test')
c = 1/math.sqrt(2*math.pi)
correct_predictions = 0
predictions = 0
for i in range(10):             # for each digit
    for im in images[i]:
        # im is an image of digit i.
        prob = [1.0] * 10
        for digit in range(10):
            prob[digit] *= py[digit]
            for j in range(256):
                prob[digit] *= c / sigma[digit][j] * math.exp(-0.5*((im[j]-mu[digit][j]) / sigma[digit][j])**2)
        k = prob.index(max(prob))
        if k == i:
            correct_predictions += 1
        predictions += 1
print('accuracy =', float(correct_predictions) / predictions)
from PIL import Image
import numpy as np

import sys
sys.path.append('..')
#import clusterizer


#circuit = clusterizer.circuit.MergedCircuit(3010)
#circuit.build()

im = Image.open('vragen.png')
rgb_im = im.convert('RGB')

w, h = im.width, im.height

Nimage = 1000000

noise_prob = .01

noise_mean = 700
image_mean = 10000
image_std = 3000

s = []

length = 4100
timestart = np.datetime64("2018-08", "ns")
timeend = np.datetime64("2019-02", "ns")

xscale = length / w

for i in range(Nimage):
    y = (h*i)//Nimage
    x = int(np.random.uniform(w))
    if rgb_im.getpixel((x, y))[0] < 127:
        a, b = np.random.normal(0, .5, 2)

        x, y = x+a, y+b

        x *= xscale
        y = (1.0-y/h) * (timeend - timestart) + timestart

        timestr = np.datetime_as_string(y, unit="s").replace("T", " ")

        charge = np.random.normal(image_mean, image_std)

        s.append("{0};{1};{2}".format(timestr, x, charge))
    y = (h*i)//Nimage
    x = int(np.random.uniform(w))
    if np.random.uniform() < noise_prob:
        a, b = np.random.normal(0, .5, 2)

        x, y = x+a, y+b

        x *= xscale
        y = (y/h) * (timeend - timestart) + timestart

        timestr = np.datetime_as_string(y, unit="s").replace("T", " ")

        charge = np.random.exponential(noise_mean)

        s.append("{0};{1};{2}".format(timestr, x, charge))


# Overrides the draw function to use smaller dots
output = "Date/time (UTC);Location in meters (m);Charge (picocoulomb)\n"

output += "\n".join(s)

print(output)

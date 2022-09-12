import sys, csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""2 Photon Data Acquisition - A method to compile a csv with the relevant data from screenshots
   function: python 2P_DA.py <individual_2P_scan>"""

dataTypes = ['Width','Height','Wavelength','Start','End','Stack/TimeSeries','Current Position']
data = []

filename = sys.argv[1]
prefix = filename[0:filename.find('PMT')]

# Try to open a jpg forit
try:
    imgName = prefix + '.JPG'
    img = mpimg.imread(imgName)
    imgplot = plt.imshow(img)
    plt.show(block=False)
except:
    print("There is no nicely named jpeg for this. Find it yourself")

print(prefix + '\n')
for dataType in dataTypes:
    data.append(input(dataType + ': '))

csvName = prefix + '.csv'
with open(csvName, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(dataTypes)
    writer.writerow(data)

plt.close('all')
f.close()
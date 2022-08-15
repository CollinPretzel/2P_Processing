import sys, csv

"""2 Photon Data Acquisition - A method to compile a csv with the relevant data from screenshots
   function: python 2P_DA.py <individual_2P_scan>"""

dataTypes = ['Width','Height','Wavelength','Start','End','Stack/TimeSeries','Current Position']
data = []

filename = sys.argv[1]
prefix = filename[0:filename.find('_PMT -')]

print(prefix + '\n')
for dataType in dataTypes:
    data.append(input(dataType + ': '))

csvName = prefix + '.csv'
with open(csvName, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(dataTypes)
    writer.writerow(data)

f.close()
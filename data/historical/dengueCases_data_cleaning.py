import matplotlib.pyplot as plt
import pandas 
import numpy

column_names = ['Date', 'Count']
mdata = pandas.read_csv('data/historical/csv_files/dengue_cases.csv', names=column_names, header=0)
mdata.head()

mdata.interpolate()
mdata.to_csv('data/historical/csv_files/dengue_cleaned.csv', index=False)

#*=====================================*
#*   data visualization for testing    *
#*=====================================*

#rainfall
mdata['Count'].plot()
plt.show()
import matplotlib.pyplot as plt
import pandas 
import numpy

# reading csv files
wdata2024 = pandas.read_csv('data/historical/csv_files/with2024_nosearches.csv', names=['date','RAINFALL','TMEAN','RH','searches1','searches2','Cases'], header=0)
wdata2024.head()

wdatanew = wdata2024.drop(columns=['searches1', 'searches2'])

wdatanew.to_csv('data/historical/csv_files/with2024_ver1.csv', index=False)
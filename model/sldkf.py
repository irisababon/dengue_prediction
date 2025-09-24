import matplotlib.pyplot as plt

# data
keywords = ['"dengue"', '"dengue symptoms"', '"sintomas ng dengue"', '"dengue sintomas"', '"mosquito"', '"lamok"']
correlations = [0.7433, 0.6996, 0.6953, 0.6491, 0.5001, 0.1674]

years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
means = [7.193548387, 17.19354839, 17.22580645, 10.22580645, 8.64516129, 12.96774194, 10.61290323, 16.70967742, 23.96774194, 28.22580645, 24.74193548, 5.225806452, 2.709677419, 9.258064516, 11.96774194, 35.40740741]

list1 = [28.34388501, 16.7623991 , 34.5006248 , 40.90708588, 31.22342327,
       23.3664665 , 27.19854506, 17.32776534, 33.48030578, 39.8956515 ,
       30.88114014, 23.75625936, 26.53324056, 18.07832182, 32.24234093,
       38.45805537, 30.41488536, 24.06371159, 25.95586813, 18.9191454 ,
       30.93804953, 36.82473774, 29.86273398, 24.25439342, 25.45257355,
       19.76743475, 29.66225487, 35.14103758, 29.26091301, 24.3244949]

print(sum(list1)/len(list1))

# create bar graph
plt.figure()
plt.plot(years, means, marker='o')
plt.xlabel('Year')
# plt.hlines(y=[0.7], xmin=-0.37, xmax=5.5, colors=['red'], linestyles=['--'], linewidth=1.5)
# plt.text(3.8, 0.75, 'Positive relationship', ha='left', va='center', color='red')
plt.ylabel("Mean Dengue Case Count")
plt.title("January Mean Dengue Case Counts")
# plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# plt.gca().set_ylim([0, 1])
plt.show()
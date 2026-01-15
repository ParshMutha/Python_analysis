import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

d1 = pd.read_csv("nhanes_adult_female_bmx_2020.csv")
d2 = pd.read_csv("nhanes_adult_male_bmx_2020.csv")
d1.head()
d2.head()

d1.info()
d2.info()


'''On a single plot (use matplotlib.pytplot.subplot), draw two histograms: for female 
weights (top subfigure). and for male weights (bottom subfigure) Call 
matplotlib.pyplot.xlim to make the xaxis limits identical for both subfigures (work out the 
appropriate limits yourself). '''


d1 = d1["BMXWT"].dropna()
d2 = d2["BMXWT"].dropna()

# Determine common x-axis limits
xmin = min(d1.min(), d2.min())
xmax = max(d1.max(), d2.max())

# Create figure
plt.figure(figsize=(8, 8))

# Top subplot: Female weights
plt.subplot(2, 1, 1)
plt.hist(d1, bins=20)
plt.title("Histogram of Female Weights")
plt.xlabel("Weight (kg)")
plt.ylabel("Frequency")
plt.xlim(xmin, xmax)

# Bottom subplot: Male weights
plt.subplot(2, 1, 2)
plt.hist(d2, bins=20)
plt.title("Histogram of Male Weights")
plt.xlabel("Weight (kg)")
plt.ylabel("Frequency")
plt.xlim(xmin, xmax)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

'''4. Call matplotlib.pyplot.boxplot to draw a box-and-whisker plot, with two boxes side by 
side, giving the male and female weights so that they can be compared to each other. 
Note that the boxplot function can be fed with a list of two vectors like [female_weights, 
male_weights]. In your own words, discuss the results. '''


plt.figure(figsize=(6, 6))
plt.boxplot([d1, d2], labels=["Female", "Male"])
plt.title("Box-and-Whisker Plot of Adult Weights")
plt.ylabel("Weight (kg)")
plt.show()


'''To the female matrix, add the eight column which gives the 
body mass indices of all 
the female participants.'''

d1["BMI"] = d1["BMXWT"] / ((d1["BMXHT"] / 100) ** 2)

d1.head()

''' Create a new matrix zfemale being a version of the female dataset with all its columns 
standardised (by computing the z-scores of each column). '''
d1["BMI"] = d1["BMXWT"] / ((d1["BMXHT"] / 100) ** 2)

# Create standardised female matrix using z-scores
zfemale = (d1 - d1.mean()) / d1.std()

# View first few rows
zfemale.head()

'''Draw a scatterplot matrix (pairplot) for the standardised versions of height, weight, 
waist circumference, hip circumference, and BMI of the females (based on zfemale). 
Compute Pearson’s and Spearman’s correlation coefficients for all pairs of variables. 
Interpret the obtained results.'''

female = d1[['BMXHT', 'BMXWT', 'BMXWAIST', 'BMXHIP']].dropna()
female.columns = ['Height', 'Weight', 'Waist', 'Hip']


female['BMI'] = female['Weight'] / ((female['Height'] / 100) ** 2)


zfemale = (female - female.mean()) / female.std()

# Scatterplot matrix (pairplot)
scatter_matrix(zfemale, figsize=(10, 10), diagonal='hist')
plt.suptitle("Scatterplot Matrix (Standardized Female Data)", y=1.02)
plt.show()

# Pearson correlation coefficients
pearson_corr = zfemale.corr(method='pearson')
print("Pearson Correlation Matrix:\n", pearson_corr)

# Spearman correlation coefficients
spearman_corr = zfemale.corr(method='spearman')
print("\nSpearman Correlation Matrix:\n", spearman_corr)


'''Compute the waist circumference to height ratio and the waist circumference to hip 
circumference ratio of the male and female participants by adding two more columns to 
the males and females matrices.'''


female.columns = ['Height', 'Waist', 'Hip']
male.columns   = ['Height', 'Waist', 'Hip']


female['Height_m'] = female['Height'] / 100
male['Height_m']   = male['Height'] / 100

# Compute ratios
female['Waist_to_Height_Ratio'] = female['Waist'] / female['Height_m']
female['Waist_to_Hip_Ratio']    = female['Waist'] / female['Hip']

male['Waist_to_Height_Ratio'] = male['Waist'] / male['Height_m']
male['Waist_to_Hip_Ratio']    = male['Waist'] / male['Hip']

# Drop intermediate height column if not needed
female = female.drop(columns=['Height_m'])
male   = male.drop(columns=['Height_m'])

# Display first few rows
print("Female dataset with ratios:\n", female.head())
print("\nMale dataset with ratios:\n", male.head())

''' Draw a box-and-whisker plot with four boxes side by side, comparing the distribution 
of the waistto-height ratio and the waist-to-hip ratio of both male and female participants. 
Explain what you see. '''

# Convert height from cm to meters
female['Height_m'] = female['Height'] / 100
male['Height_m']   = male['Height'] / 100

# Compute ratios
female_wht = female['Waist'] / female['Height_m']
female_whr = female['Waist'] / female['Hip']

male_wht = male['Waist'] / male['Height_m']
male_whr = male['Waist'] / male['Hip']

# Box-and-whisker plot
plt.figure()
plt.boxplot(
    [female_wht, female_whr, male_wht, male_whr],
    labels=[
        'Female Waist-to-Height',
        'Female Waist-to-Hip',
        'Male Waist-to-Height',
        'Male Waist-to-Hip'
    ]
)
plt.ylabel("Ratio Value")
plt.title("Waist-to-Height and Waist-to-Hip Ratio Comparison")
plt.show()

'''Print out the standardised body measurements for the 5 persons with the lowest BMI 
and the 5 persons with the 5 highest BMI (e.g., call print for a subset of zfemale comprised 
of 10 chosen rows as determined by a call to numpy.argsort). Interpret the results.'''



female = d1[['BMXHT', 'BMXWT', 'BMXWAIST', 'BMXHIP']].dropna()
female.columns = ['Height', 'Weight', 'Waist', 'Hip']

# Compute BMI
female['BMI'] = female['Weight'] / ((female['Height'] / 100) ** 2)

# Standardize (z-scores)
zfemale = (female - female.mean()) / female.std()

# Get indices of 5 lowest and 5 highest BMI values
bmi_sorted_idx = np.argsort(zfemale['BMI'].values)

lowest_5 = bmi_sorted_idx[:5]
highest_5 = bmi_sorted_idx[-5:]

# Combine indices
selected_idx = np.concatenate([lowest_5, highest_5])

# Print standardized measurements
print(zfemale.iloc[selected_idx])

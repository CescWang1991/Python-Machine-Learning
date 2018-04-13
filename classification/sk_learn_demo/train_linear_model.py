import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from classification.function import prepare_country_stats


# Load the data
oecd_data = pd.read_csv("../data/oecd_bli_2015.csv", thousands=',')
gdp_data = pd.read_csv("../data/gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_data, gdp_data)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = linear_model.LinearRegression()

# Train the model
model.fit(x, y)

# Make a prediction for Cyprus
x_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(x_new))

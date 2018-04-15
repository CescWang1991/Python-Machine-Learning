# Case: house price predication

## Introduction
This case aim to build a model of housing prices in California using the California census data. The [California Housing Prices Dataset](../data/housing.csv) comes from: </br>
* R. Kelley Pace and Ronald Barry, “Sparse Spatial Autoregressions,” Statistics & Probability Letters 33, no. 3 (1997): 291–297.</br>

This data has metrics such as the population, median income, median housing price, and so on for each block group in California. Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data.

## Notebook
The full Jupyter notebook is available at [handson-ml](https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb).

## Data Dictionary

| Feature Name | Description | Type |
| :----- | :----- | :----- |
| longitude | longitude of district | numerical value |
| latitude | latitude of district | numerical value |
| housing_median_age | median age of house in district | numerical value |
| total_rooms | number of total rooms in district | numerical value |
| total_bedrooms | number of total bedrooms in district | numerical value |
| population | district's population | numerical value |
| households | number of house holds in district | numerical value |
| median_income | median income of resident | numerical value |
| ocean_proximity | <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND | categorical value |

| Label Name | Description | Type |
| :----- | :----- | :----- |
| median_house_value | median house value | numerical value |

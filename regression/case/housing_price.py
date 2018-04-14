import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

path = "../data/housing.csv"
housing_data = pd.read_csv(path)

# print(housing_data.head()) #take a look at the top five rows using the DataFrame’s head() method

# print(housing_data.info()) #get a quick description of the data

# print(housing_data["ocean_proximity"].value_counts())
# find out what categories exist and how many districts belong to each category

# print(housing_data.describe()) shows a summary of the numerical attributes

# housing_data.hist(bins=50, figsize=(20, 15))
# plt.show()
# call the hist() method on the whole dataset, and it will plot a histogram for each numerical attribute


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# np.random.seed(42)  # Let it always generates the same shuffled indices
# train_set, test_set = split_train_test(housing_data, 0.2)
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

# Divide by 1.5 to limit the number of income categories
housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
# Label those above 5 as 5
housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = strat_train_set.copy()
# create a scatterplot of all districts to visualize the data
# makes it much easier to visualize the places where there is a high density of data points
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100,
#              label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# plots every numerical attribute against every other numerical attribute
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
new_corr_matrix = housing.corr()
# print(new_corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# the total_bedrooms attribute has some missing values
# housing.dropna(subset=["total_bedrooms"])   # Get rid of the corresponding districts.
# housing.drop("total_bedrooms", axis=1)      # Get rid of the whole attribute.
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median)    # Set the values to some value (zero, the mean, the median, etc.).
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
# print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# transform categorical attributes
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

# apply both transformations
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

# DATA LOADING
# Download the dataset only if it is not already available locally,
# then extract the archive and return the CSV as a pandas DataFrame.
def load_housing_data():
    # Local path used to store the compressed dataset.
    tarball_path = Path("datasets/housing.tgz")

    # Download the archive only if it is not already present.
    # This avoids downloading the dataset at every run.
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)

    # Extract the .tgz archive into the datasets directory.
    # The CSV will then be available at datasets/housing/housing.csv.
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")

    # Load the CSV with pandas and return it as a DataFrame.
    # The returned object can now be explored with standard pandas methods such as:
    # housing.info(), housing.describe(), and housing.head().
    return pd.read_csv(Path("datasets/housing/housing.csv"))


# Load the dataset into the housing DataFrame.
housing = load_housing_data()


# DATA EXPLORATION
"""
print(housing.head())
# Display the first five rows.

print(housing.info())
# Display a DataFrame summary: number of rows, columns, non-null values,
# and the data type of each column.

print(housing["ocean_proximity"].value_counts())
# Count the observations in each ocean_proximity category.

print(housing.describe())
# Display descriptive statistics for numerical features: mean, standard deviation,
# minimum, maximum, and quartiles.
"""


import matplotlib.pyplot as plt

# Plot settings used throughout the notebook/script.
plt.rc('font', size=8)
plt.rc('axes', labelsize=8, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=10)


# Histograms of the numerical features
"""
housing.hist(bins=50, figsize=(12, 8))
plt.show()
"""
# housing.hist() creates one histogram for each numerical feature.
# bins=50 divides each feature range into 50 intervals.
# This helps reveal distributions, outliers, different scales,
# and possible artificial limits in the data.


import numpy as np

# Manual train/test split using shuffled row indices
"""
def shuffle_and_split_data(data, test_ratio):
    # data: DataFrame to split.
    # test_ratio: fraction of the dataset assigned to the test set.
    # Example: test_ratio=0.2 means 20% test data and 80% training data.

    # Set a seed to make the split reproducible.
    # With the same dataset and row order, this produces the same split.
    np.random.seed(42)

    # Shuffle all row indices.
    shuffled_indices = np.random.permutation(len(data))

    # Compute the number of rows assigned to the test set.
    test_set_size = int(len(data) * test_ratio)

    # Use the first shuffled indices for testing and the rest for training.
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # iloc selects rows by integer position.
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = shuffle_and_split_data(housing, 0.2)

# Limitation:
# a fixed seed makes the split reproducible only while the dataset stays unchanged.
# Reordering, adding, or removing rows can change which samples fall into each set.
# A more stable solution is to assign rows using a persistent identifier
# rather than their current position.
"""


# Stable train/test split using a CRC32 hash
"""
from zlib import crc32


def is_id_in_test_set(identifier, test_ratio):
    # Return True when a row should belong to the test set.
    # The decision is based on a stable row identifier rather than row position.

    # crc32 produces a deterministic hash: the same identifier
    # always maps to the same value in the range 0 ... 2^32 - 1.
    # This gives a stable pseudo-random assignment.

    # If the hash falls within the test portion of the range, assign the row to the test set.
    # With test_ratio=0.2, roughly 20% of identifiers are assigned to the test set.
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    # id_column contains a stable identifier for each row.
    ids = data[id_column]

    # apply evaluates the test-set rule for every identifier.
    # The lambda takes one identifier
    # and returns True when it belongs to the test set.
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))

    # data.loc[in_test_set] selects test rows.
    # data.loc[~in_test_set] selects the complementary training rows.
    return data.loc[~in_test_set], data.loc[in_test_set]


# The dataset does not provide a dedicated ID column.
# Use the row index as a temporary identifier.
# This remains stable only if new rows are appended
# and the order of existing rows does not change.
housing_with_id = housing.reset_index()

train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
"""


# Train/test split with scikit-learn
"""
from sklearn.model_selection import train_test_split

# train_test_split automatically creates training and test sets.
# random_state=42 makes the split reproducible while the dataset remains unchanged.
# If the dataset changes or is reordered, some rows may still move
# between the training and test sets.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
"""


# STRATIFIED SAMPLING
# median_income is strongly related to median_house_value.
# Group income values into categories and use them for stratified sampling
# so that training and test sets preserve similar proportions
# of low-, medium-, and high-income districts.
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Inspect the income-category distribution
"""
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
"""
# The bar chart shows how many districts fall into each income category.
# It helps identify strongly imbalanced categories.


from sklearn.model_selection import StratifiedShuffleSplit

# Alternative stratified split with StratifiedShuffleSplit
"""
# StratifiedShuffleSplit can generate multiple train/test splits.
# n_splits=10 generates 10 different splits.
# test_size=0.2 assigns 20% of the rows to each test set.
# Stratification is based on housing["income_cat"].
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# Use the first generated split.
strat_train_set, strat_test_set = strat_splits[0]
"""


from sklearn.model_selection import train_test_split

# More compact way to create a stratified split directly.
# stratify=housing["income_cat"] preserves approximately the same income-category
# proportions in the training set, test set, and full dataset.
strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42,
    stratify=housing["income_cat"]
)

# Compare income-category proportions across the full, training, and test sets
"""
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(housing["income_cat"].value_counts() / len(housing))
"""
# If stratification worked correctly, category proportions should be very similar
# across all three outputs.


# Remove income_cat because it was created only for stratified sampling.
# It should not be used as an artificial model feature.
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# From this point onward, work only with a copy of the training set.
# The test set must remain untouched during exploration and preprocessing
# until the final model evaluation.
housing = strat_train_set.copy()


# GEOGRAPHICAL VISUALIZATION
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
# plt.show()

# alpha=0.2 makes points semi-transparent so overlapping observations
# reveal denser areas more clearly.
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# plt.show()


# Geographic view with population and median house value
"""
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    s=housing["population"] / 100,   # point size represents population
    label="population",
    c="median_house_value",          # point color represents median house value
    cmap="plasma",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7)
)
plt.show()
"""
# This plot combines:
# - district location;
# - population through point size;
# - median house value through point color.


# CORRELATION ANALYSIS
# corr() computes Pearson correlation coefficients between numerical features.
# Values close to 1 indicate a strong positive linear relationship;
# values close to -1 indicate a strong negative linear relationship;
# values close to 0 indicate little or no linear relationship.
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)


from pandas.plotting import scatter_matrix

# Scatter matrix for a few important numerical features.
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))


# median_income shows the clearest relationship with median_house_value,
# so inspect it with a more detailed scatter plot.
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)

# FEATURE ENGINEERING
# Some raw features become more informative when combined.
# For example, total_rooms depends heavily on district size, while
# rooms_per_house captures the average size of a household more directly.
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["people_per_house"] = housing["population"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]

# Recompute correlations after adding the engineered features.
# A stronger absolute correlation with median_house_value may indicate
# that a new feature is more useful to the model.
# Correlation captures linear relationships only and does not imply causation.
corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


# Separate input features from the target labels.
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# print(housing.info())
# print(housing_labels.info())


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# Keep only numerical features for the numerical preprocessing examples.
housing_num = housing.select_dtypes(include=[np.number])

# fit() learns the median of each numerical feature.
imputer.fit(housing_num)

# transform() replaces missing values with the learned medians.
X = imputer.transform(housing_num)

# Convert the NumPy array returned by transform() back into a pandas DataFrame.
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

# OneHotEncoder creates one binary feature for each ocean_proximity category.
# A row contains 1 for its category and 0 for the others.
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Numerical features have very different scales.
# For example, total_rooms spans a much wider range than median_income.
# Min-max scaling maps features to a common range before modeling.

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

housing_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# Standardization is another common way to put numerical features on comparable scales.
# Unlike MinMaxScaler, it does not constrain values to a fixed interval.
# It is also less affected by extreme values than min-max scaling.

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

housing_num_std_scaled = std_scaler.fit_transform(housing_num)



# Apply a logarithmic transformation and define exp() as its inverse.

from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)

log_pop = log_transformer.transform(housing["population"])


# Use an RBF kernel to transform values into similarity scores.
from sklearn.metrics.pairwise import rbf_kernel

# Compute the similarity between housing_median_age and the reference value 35.
# gamma controls how quickly similarity decreases as values move away from 35.
rbf_transformer = FunctionTransformer(rbf_kernel,
kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])


# The same result can be obtained by calling rbf_kernel directly.
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# Simplified custom implementation of StandardScaler.
# fit() learns the mean and standard deviation; transform() standardizes the data.
class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean
    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!
    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    

from sklearn.cluster import KMeans

# Learn geographic cluster centers with KMeans, then transform
# latitude and longitude into RBF similarities to each cluster center.
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# Create 10 geographic clusters from latitude and longitude.
# Weight districts by median_house_value when fitting the cluster centers.
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                           sample_weight=housing_labels)

    
# Pipelines apply multiple preprocessing steps sequentially.
# Each step receives the output of the previous one.
# Here, missing values are first imputed with the median and then standardized.
# Additional transformers can be inserted when needed.

from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median" ), StandardScaler())


# ColumnTransformer applies different preprocessing pipelines to different feature groups.
# This lets numerical and categorical features be handled separately.
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms","total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

# Each tuple defines a name, a transformer, and the columns it should process.
preprocessing = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)])


# Column selectors can choose features automatically based on dtype,
# avoiding the need to list every column manually.

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

# Apply the preprocessing pipeline to the housing data.

housing_prepared = preprocessing.fit_transform(housing)



# FINAL PREPROCESSING PIPELINE

# Compute the ratio between the two input columns.
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

# Name the feature created by FunctionTransformer.
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

# Impute missing values, compute the ratio, then standardize it.
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

# Apply a log transform to long-tailed features, then standardize them.
# one-to-one preserves the input feature names.
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# Convert latitude and longitude into similarities to 10 geographic clusters.
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

# Default preprocessing for numerical features without a dedicated transformation.
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

# Apply specialized transformations to different feature groups.
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age 


# MODEL TRAINING AND EVALUATION
# Start with a Linear Regression baseline.
from sklearn.linear_model import LinearRegression

# Combine preprocessing and the estimator into a single pipeline.
lin_reg = make_pipeline(preprocessing, LinearRegression())

# Fit the pipeline on the training features and labels.
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)

# Compare the first five predictions with the true values.
#   print( housing_predictions[:5].round(-2) )
#   print( housing_labels[:5].round(-2))

# Evaluate the training error with RMSE. The relatively high value suggests
# that Linear Regression underfits this dataset.
from sklearn.metrics import root_mean_squared_error

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
#   print(lin_rmse)

# Try a more flexible DecisionTreeRegressor.
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing , housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)

#   print(tree_rmse)
# A training RMSE of 0 is a warning sign of severe overfitting, not proof of perfect generalization.


# Use cross-validation for a more reliable estimate of model performance.
# With 10-fold cross-validation, the model trains on 9 folds and validates on the remaining fold,
# repeating the process until every fold has been used for validation once.

from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

# print(pd.Series(tree_rmses).describe())

# Cross-validation reveals a non-zero validation error because the Decision Tree
# can no longer be evaluated on the exact same data it memorized during training.
# Each fold evaluates the model on data excluded from that fold's training step.


# Try a Random Forest, which combines many Decision Trees into an ensemble.

from sklearn.ensemble import RandomForestRegressor

# forest_reg = make_pipeline(preprocessing , RandomForestRegressor(random_state = 42, n_jobs = 1))

# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring ="neg_root_mean_squared_error", cv = 3, verbose=2 )

#   print(pd.Series(forest_rmses).describe())


# Random Forest usually performs better than the previous baseline models.
# It may still overfit, so the next step is hyperparameter tuning.


# Use GridSearchCV to evaluate predefined hyperparameter combinations
# instead of tuning them manually.

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
# Tune both preprocessing and model hyperparameters: n_clusters and max_features.

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error', verbose=2)
#   grid_search.fit(housing, housing_labels)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# RANDOMIZED SEARCH
# Search for a good hyperparameter combination by sampling random values
# from the specified distributions. Unlike Grid Search, it does not evaluate
# every possible combination; n_iter controls the number of sampled trials.

# Distributions used to sample hyperparameter values.
# randint includes low and excludes high: n_clusters ranges from 3 to 49,
# while max_features ranges from 2 to 19.
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20)
}


# Evaluate 10 randomly sampled hyperparameter combinations.
# Each combination is evaluated using 3-fold cross-validation.
rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,                              # number of sampled combinations
    cv=3,                                   # 3-fold cross-validation
    scoring="neg_root_mean_squared_error",  # optimize the negative RMSE score
    random_state=42                         # make random sampling reproducible
)

# Run the search: 10 combinations x 3 folds = 30 fits.
rnd_search.fit(housing, housing_labels)

# Keep the best complete pipeline found by RandomizedSearchCV.
final_model = rnd_search.best_estimator_  # includes preprocessing


# FINAL TEST SET EVALUATION
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)

# print(final_rmse)

import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")
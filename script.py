# Step 1: Deciding (not) to Segment
def decide_to_segment(market_power, heterogeneous=True):
    if market_power and not heterogeneous:
        return "Cater to entire market, no need to segment."
    else:
        return "Investigate market heterogeneity, use a differentiated strategy."

# Step 2: Specifying the Ideal Target Segment
def is_attractive_segment(homogeneous, distinct, large_enough, matching_strengths, identifiable, reachable):
    if all([homogeneous, distinct, large_enough, matching_strengths, identifiable, reachable]):
        return "Segment is attractive."
    else:
        return "Segment is not attractive."

def target_segment_criteria(likes_mcdonalds, eats_frequently, current_share, growth_potential=False):
    if likes_mcdonalds and eats_frequently:
        return "Target segment is already favorable."
    elif growth_potential:
        return "Potential growth segment, modify perceptions."
    else:
        return "Not a favorable segment."

# Step 3: Collecting Data (Sample dataset for demonstration)
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data for respondents (replace with real data in practice)
data = {
    "YUMMY": [1, 0, 1, 0, 1],
    "CONVENIENT": [1, 1, 1, 0, 0],
    "SPICY": [0, 0, 1, 0, 0],
    "FATTENING": [1, 1, 1, 1, 1],
    "GREASY": [1, 1, 0, 1, 1],
    "FAST": [1, 1, 1, 1, 0],
    "CHEAP": [1, 1, 1, 0, 0],
    "TASTY": [1, 1, 1, 0, 0],
    "EXPENSIVE": [0, 0, 1, 0, 1],
    "HEALTHY": [0, 0, 0, 0, 0],
    "DISGUSTING": [0, 1, 0, 1, 1],
    "AGE": [25, 34, 45, 23, 56],
    "GENDER": ['Male', 'Female', 'Male', 'Female', 'Female']
}

df = pd.DataFrame(data)

def summarize_data(df):
    print("Summary of McDonald's Brand Image Data:")
    print(df.describe())

summarize_data(df)

# Step 4: Exploring Data
# Simulated dataset for PCA analysis
data = {
    "yummy": ['No', 'Yes', 'No', 'Yes', 'No'],
    "convenient": ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    "spicy": ['No', 'No', 'Yes', 'No', 'Yes'],
    "fattening": ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    "greasy": ['No', 'Yes', 'Yes', 'No', 'Yes'],
    "fast": ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
    "cheap": ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    "tasty": ['No', 'Yes', 'Yes', 'No', 'Yes'],
    "expensive": ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    "healthy": ['No', 'No', 'Yes', 'No', 'Yes'],
    "disgusting": ['No', 'No', 'Yes', 'Yes', 'No'],
    "Like": [-3, 2, 1, -1, 2],
    "Age": [61, 51, 62, 45, 53],
    "VisitFrequency": ["Every three months"] * 5,
    "Gender": ['Female', 'Female', 'Female', 'Male', 'Male']
}

df = pd.DataFrame(data)

# Basic inspection of the dataset
print("Column Names:", df.columns)
print("Dataset Dimensions:", df.shape)
print("First 3 rows:\n", df.head(3))

# Converting Yes/No responses to binary values
segmentation_columns = ["yummy", "convenient", "spicy", "fattening", "greasy", "fast", 
                        "cheap", "tasty", "expensive", "healthy", "disgusting"]
MD_x = df[segmentation_columns].applymap(lambda x: 1 if x == "Yes" else 0)
print("Average Values of Segmentation Variables:\n", MD_x.mean().round(2))

# Step 4c: Principal Component Analysis (PCA)
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Output variance explained by each principal component
explained_variance = pca.explained_variance_ratio_
print("Variance explained by each component:", np.round(explained_variance, 4))
print("Cumulative variance explained:", np.cumsum(explained_variance))

# Step 4d: Display PCA loadings (the principal component weights)
num_components = MD_pca.shape[1]  # Get the number of components from the PCA transformation
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(num_components)], index=segmentation_columns)
print(loadings)


# Step 4e: Plot the perceptual map (first two principal components)
plt.figure(figsize=(10, 7))
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey', alpha=0.5)
for i, (var, pc1, pc2) in enumerate(zip(segmentation_columns, loadings['PC1'], loadings['PC2'])):
    plt.arrow(0, 0, pc1, pc2, color='r', alpha=0.5)
    plt.text(pc1*1.15, pc2*1.15, var, color='r', ha='center', va='center', fontsize=10)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title("Perceptual Map (PCA)")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

# Step 5: Clustering Analysis
from flexclust import stepFlexclust, bootFlexclust, relabel, slswFlexclust
import random
random.seed(1234)

# Clustering for 2 to 8 clusters
MD_km28 = stepFlexclust(MD_x, 2, 8, nrep=10, verbose=False)
MD_km28 = relabel(MD_km28)

# Bootstrap clustering analysis
MD_b28 = bootFlexclust(MD_x, 2, 8, nrep=10, nboot=100)

# Select 4 clusters for further analysis
MD_k4 = MD_km28["4"]
MD_r4 = slswFlexclust(MD_x, MD_k4)

# Plot cluster stability
plt.plot(MD_r4, ylim=[0, 1], xlab="Segment Number", ylab="Segment Stability")
plt.show()

# Step 6: Regression Model using Flexmix
from flexmix import FLXMCmvbinary, stepFlexmix

# Regression formula
formula = f"Like.n ~ {' + '.join(segmentation_columns)}"

# Fitting a 2-segment model
MD_reg2 = stepFlexmix(formula, data=df, k=2, nrep=10, verbose=False)

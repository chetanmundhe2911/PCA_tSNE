# Fit PCA and analyze explained variance
pca = PCA().fit(scaled_data)
explained_variance = pca.explained_variance_ratio_.cumsum()

# Find the minimum number of components that explain at least 90% of the variance
n_components_optimal = next(i for i, cumulative_variance in enumerate(explained_variance) if cumulative_variance >= 0.80)

print(f"Optimal number of components to retain 90% variance: {n_components_optimal}")

# Apply PCA with the optimal number of components
pca = PCA(n_components=n_components_optimal)
pca_data = pca.fit_transform(scaled_data)


#Output
# Optimal number of components to retain 90% variance: 144

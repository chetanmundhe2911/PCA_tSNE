import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the dataset
file_path = '/home/ec2-user/python/modeltraining/pca/train.csv'
dataset = pd.read_csv(file_path)

# Step 1: Preprocess the Data
# Apply one-hot encoding to categorical columns
encoded_data = pd.get_dummies(dataset, columns=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])

# Separate the target variable 'y' and features
y = dataset['y']
X = encoded_data.drop(columns=['y'])

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Optional: Use PCA for initial dimensionality reduction
pca = PCA(n_components=30)
pca_data = pca.fit_transform(scaled_data)

# Step 2: Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
tsne_result = tsne.fit_transform(pca_data)

# Visualize the Results using plt.scatter for color mapping
plt.figure(figsize=(10, 8))
sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap='viridis')
plt.colorbar(sc)
plt.title("t-SNE plot of Mercedes-Benz dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels


class ClusterSimilarity(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=10, gamma=1.0, random_state=0):
        self.kmeans_ = KMeans()
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, x, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(x, sample_weight=sample_weight)

        return self

    def transform(self, x):
        return pairwise_kernels(x, self.kmeans_.cluster_centers_, metric='rbf', gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f'Cluster {i} similarity' for i in range(self.n_clusters)]


if __name__ == '__main__':
    housing = pd.read_csv('../files/housing/housing.csv')
    housing_labeled = housing[['median_house_value']].to_numpy().reshape(1, -1)[0]
    print(housing_labeled)
    print(housing.describe())

    customKmeansClustering = ClusterSimilarity(n_clusters=5, gamma=1, random_state=42)
    similarities = customKmeansClustering.fit_transform(housing[['latitude', 'longitude']],
                                                        sample_weight=housing_labeled)
    print(similarities[:3].round(2))

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

    housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                         s=housing_renamed["Population"] / 100, label="Population",
                         c="Max cluster similarity",
                         cmap="jet", colorbar=True,
                         legend=True, sharex=False, figsize=(10, 7))

    plt.plot(customKmeansClustering.kmeans_.cluster_centers_[:, 1],
             customKmeansClustering.kmeans_.cluster_centers_[:, 0],
             linestyle="", color="black", marker="X", markersize=20,
             label="Cluster centers")
    plt.legend(loc="upper right")
    plt.savefig('../Figures/latitudeLongitudeHousingAfinities.png')
    plt.show()

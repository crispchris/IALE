import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans

from active.strategy import Strategy


class KMeansSampling(Strategy):
    def __init__(self, dataset_pool, idxs_lb):
        super(KMeansSampling, self).__init__(dataset_pool, idxs_lb)

    def query(self, n, model, options):
        n_pool = len(self.dataset_pool.y)
        idxs_unlabeled = np.arange(n_pool)[~self.idxs_lb]

        # Get embeddings for unlabelled samples
        pool_handler = self.get_handler("pool")
        batch_size = max(100, int(len(pool_handler) / 100))
        pool_sampler = torch.utils.data.SequentialSampler(pool_handler)
        pool_loader = torch.utils.data.DataLoader(pool_handler, sampler=pool_sampler, batch_size=batch_size)
        embeddings = self.get_embeddings(model, pool_loader)
        embeddings = embeddings

        # Perform KMeans on embeddings
        print("Performing KMeans on embeddings...")
        kmeans = KMeans(n_clusters=n, random_state=0).fit(embeddings)
        # Using MiniBatchKMeans to improve performance - introduces more variance due to Stochastic GD
        # kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, batch_size=int(n*1.5)).fit(embeddings)

        print("Predicting clusters...")
        cluster_idxs = kmeans.predict(embeddings)
        centers = kmeans.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)

        q_idxs = []
        embed_range = np.arange(embeddings.shape[0])
        for i in range(n):
            current_dis = dis[cluster_idxs == i]
            if current_dis.size > 0:
                q_idxs.append(embed_range[cluster_idxs == i][current_dis.argmin()])
        q_idxs = np.array(q_idxs)

        return idxs_unlabeled[q_idxs]

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_excel('Online Retail.xlsx')


df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalSum'] = df['Quantity'] * df['UnitPrice']


snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSum': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']


scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])


pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)


kmeans = KMeans(n_clusters=4, random_state=42)
rfm['KMeansCluster'] = kmeans.fit_predict(rfm_scaled)


dbscan = DBSCAN(eps=0.8, min_samples=10)
rfm['DBSCANCluster'] = dbscan.fit_predict(rfm_scaled)


linked = linkage(rfm_scaled, method='ward')
rfm['HierarchicalCluster'] = fcluster(linked, 4, criterion='maxclust') - 1

plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['KMeansCluster'], cmap='viridis', s=10)
plt.title('KMeansCluster')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 3, 2)
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['DBSCANCluster'], cmap='viridis', s=10)
plt.title('DBSCANCluster')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 3, 3)
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['HierarchicalCluster'], cmap='viridis', s=10)
plt.title('HierarchicalCluster')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()


def describe_clusters(df, cluster_col):
    print(f"\nCustomer characteristics by cluster: {cluster_col}\n")

    # Calculate mean RFM values per cluster
    cluster_summary = df.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
    cluster_counts = df[cluster_col].value_counts().sort_index()
    cluster_summary['CustomerCount'] = cluster_counts

    print(cluster_summary)

    # Simple descriptive summary per cluster
    for cluster_id, row in cluster_summary.iterrows():
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']

        print(f"\nCluster {cluster_id} (Customers: {int(row['CustomerCount'])}) characteristics:")

        if recency < cluster_summary['Recency'].median():
            recency_desc = "Recently active customers"
        else:
            recency_desc = "Inactive for a while"

        if frequency > cluster_summary['Frequency'].median():
            frequency_desc = "High purchase frequency"
        else:
            frequency_desc = "Low purchase frequency"

        if monetary > cluster_summary['Monetary'].median():
            monetary_desc = "High monetary value"
        else:
            monetary_desc = "Low monetary value"

        print(f"  - {recency_desc}")
        print(f"  - {frequency_desc}")
        print(f"  - {monetary_desc}")

describe_clusters(rfm, 'KMeansCluster')
describe_clusters(rfm, 'DBSCANCluster')
describe_clusters(rfm, 'HierarchicalCluster')

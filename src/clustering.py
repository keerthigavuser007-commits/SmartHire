import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


print("=" * 60)
print("SMART HIRE - JOB CLUSTERING")
print("=" * 60)


# --------------------------------------------------
# 1. LOAD JOB DATA
# --------------------------------------------------

print("\nLoading job dataset...")

file_path = "Data/processed/jobs_clean.csv"

jobs = pd.read_csv(file_path)

print("Total jobs:", len(jobs))


# Remove missing job text
jobs = jobs.dropna(subset=["job_text"])

jobs["job_text"] = jobs["job_text"].astype(str)

jobs = jobs[
    jobs["job_text"].str.strip() != ""
]

jobs = jobs.reset_index(drop=True)

print("Valid jobs:", len(jobs))


# --------------------------------------------------
# 2. SAMPLE DATA
# --------------------------------------------------

# Clustering 123,000+ jobs directly can use a lot
# of memory, so use a representative sample.

sample_size = min(10000, len(jobs))

jobs_sample = jobs.sample(
    n=sample_size,
    random_state=42
).reset_index(drop=True)

print("Jobs used for clustering:", len(jobs_sample))


# --------------------------------------------------
# 3. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(
    jobs_sample["job_text"]
)

print("TF-IDF shape:", X.shape)


# --------------------------------------------------
# 4. TEST DIFFERENT K VALUES
# --------------------------------------------------

print("\nTesting different cluster counts...")

k_values = [3, 4, 5, 6, 7]

results = []

for k in k_values:

    print(f"\nTesting K = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    inertia = model.inertia_

    silhouette = silhouette_score(
        X,
        labels,
        sample_size=min(3000, X.shape[0]),
        random_state=42
    )

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette_score": silhouette
    })

    print("Inertia:", round(inertia, 2))
    print("Silhouette Score:", round(silhouette, 4))


# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

evaluation = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CLUSTER EVALUATION")
print("=" * 60)

print(evaluation)


# --------------------------------------------------
# 6. SELECT BEST K
# --------------------------------------------------

best_k = int(
    evaluation.loc[
        evaluation["silhouette_score"].idxmax(),
        "k"
    ]
)

print("\nSelected K:", best_k)


# --------------------------------------------------
# 7. FINAL K-MEANS MODEL
# --------------------------------------------------

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

jobs_sample["cluster"] = kmeans.fit_predict(X)


# --------------------------------------------------
# 8. SHOW CLUSTER SIZES
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER SIZES")
print("=" * 60)

print(
    jobs_sample["cluster"].value_counts().sort_index()
)


# --------------------------------------------------
# 9. SHOW SAMPLE JOBS FROM EACH CLUSTER
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE JOBS FROM EACH CLUSTER")
print("=" * 60)

for cluster_number in sorted(
    jobs_sample["cluster"].unique()
):

    print(
        f"\n--- CLUSTER {cluster_number} ---"
    )

    cluster_jobs = jobs_sample[
        jobs_sample["cluster"] == cluster_number
    ]

    for title in cluster_jobs["title"].head(10):

        print("-", title)


# --------------------------------------------------
# 10. PCA FOR VISUALIZATION
# --------------------------------------------------

print("\nCreating PCA representation...")

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(
    X.toarray()
)

jobs_sample["PCA_1"] = X_pca[:, 0]
jobs_sample["PCA_2"] = X_pca[:, 1]


# --------------------------------------------------
# 11. SAVE CLUSTERED DATA
# --------------------------------------------------

os.makedirs(
    "Data/processed",
    exist_ok=True
)

output_path = (
    "Data/processed/jobs_clustered.csv"
)

jobs_sample.to_csv(
    output_path,
    index=False
)

print("\nClustered jobs saved to:")
print(output_path)


# --------------------------------------------------
# 12. SAVE EVALUATION
# --------------------------------------------------

evaluation_path = (
    "Data/processed/cluster_evaluation.csv"
)

evaluation.to_csv(
    evaluation_path,
    index=False
)

print("\nCluster evaluation saved to:")
print(evaluation_path)


# --------------------------------------------------
# 13. COMPLETED
# --------------------------------------------------

print("\n" + "=" * 60)
print("JOB CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 60)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pickle
import os
import base64

def load_data():
    """
    loading our training data and serializing it 
    """
    file_path = "/opt/airflow/working_data/wholesale_customers.csv"
    df = pd.read_csv(file_path)
    print("dataset loaded")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii") 

def data_preprocessing(data_b64: str):
    """
    preprocesses Wholesale Customers data for K-Means clustering

    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    features = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen"
    ]

    # Validate required columns
    missing_columns = set(features) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select spending features
    clustering_data = df[features].copy()

    # Handle missing values
    clustering_data = clustering_data.dropna()

    # Scale features
    scaler = MinMaxScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    print("Data preprocessing completed")
    print(f"Processed shape: {clustering_data_scaled.shape}")

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_scaled)
    return base64.b64encode(clustering_serialized_data).decode("ascii")

def validate_data(data_b64: str):
    """
    Validates the customer dataset before preprocessing.
    Raises an exception if validation fails.
    """
    df = pickle.loads(base64.b64decode(data_b64))

    required_columns = [
        "Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df[required_columns].isna().any().any():
        raise ValueError("Dataset contains missing spending values")

    if (df[required_columns] < 0).any().any():
        raise ValueError("Dataset contains negative spending values")

    print("Data validation passed")

    return data_b64


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """

    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    max_k = min(15, len(df))
    k_values = list(range(1, max_k + 1))
    sse = []

    for k in k_values:
        model = KMeans(n_clusters=k, **kmeans_kwargs)
        model.fit(df)
        sse.append(float(model.inertia_))

    kl = KneeLocator(
        k_values,
        sse,
        curve="convex",
        direction="decreasing"
    )

    optimal_k = kl.elbow

    if optimal_k is None:
        raise ValueError("Could not determine optimal K using elbow method")

    print(f"Optimal number of clusters: {optimal_k}")

    optimal_model = KMeans(
        n_clusters=int(optimal_k),
        **kmeans_kwargs
    )

    optimal_model.fit(df)

    output_dir = "/opt/airflow/working_data/models"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(optimal_model, f)

    print(f"Model saved at: {output_path}")

    return {
        "optimal_k": int(optimal_k),
        "sse": sse,
        "k_values": k_values,
        "model_path": output_path
    }


def load_model_elbow(data_b64: str, model_results: dict):
    """
    Loads the optimal K-Means model and evaluates
    clustering quality using silhouette score.
    """

    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    model_path = model_results["model_path"]

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    cluster_labels = loaded_model.predict(df)

    if len(set(cluster_labels)) < 2:
        raise ValueError(
            "Silhouette score requires at least two clusters"
        )

    score = silhouette_score(df, cluster_labels)

    print(f"Optimal K: {model_results['optimal_k']}")
    print(f"Silhouette Score: {score:.4f}")

    return {
        "optimal_k": model_results["optimal_k"],
        "silhouette_score": float(score),
        "model_path": model_path
    }

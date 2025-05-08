import os
# Set environment variable BEFORE kagglehub loads
os.environ["KAGGLE_DATA_PROXY_CACHE_DIR"] = "/media/biprarshi/common/files/AI/kagglehub_cache"
import kagglehub
# Now download — it’ll save under this directory
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

print("Dataset path:", path)

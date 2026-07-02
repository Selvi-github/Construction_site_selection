import kagglehub

try:
    path = kagglehub.dataset_download("jayaprakashpondy/soil-image-dataset")
    print("Path to dataset files:", path)
except Exception as e:
    print("Error:", e)

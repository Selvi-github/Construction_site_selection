import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from predictor import predict_location

try:
    print("Testing predict_location...")
    result = predict_location(12.9716, 77.5946, "House", 2, {})
    print("Success!")
    print(result)
except Exception as e:
    print(f"Caught error: {e}")
    import traceback
    traceback.print_exc()

import sys
sys.path.append("C:/Users/lenmo/research_project")  # Path where rotate.pyd exists

import rotate

# Test with a sample image (32x32 RGB)
import numpy as np
img = np.random.rand(32, 32, 3).astype(np.float32)

# Call your C++ function (exact name from rotate.cpp)
rotated_img = img.copy()
rotate.rotate(rotated_img, 32, 32, 30.0)  # angle=30 degrees

print(" successful! Max value:", rotated_img.max())
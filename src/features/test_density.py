import numpy as np

from density import compute_density


# Create a simple test mask
mask = np.zeros((100, 100), dtype=np.uint8)

# Make 25% of the image foreground
mask[:50, :50] = 255


density = compute_density(mask)

print("Density :", density)
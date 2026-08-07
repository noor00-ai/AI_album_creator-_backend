import cv2
import numpy as np


class FeatureExtractor:
    """
    Extracts image features using the same preprocessing
    as the clustering notebook.
    """

    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size

    def extract(self, image_path):
        """
        Extract feature vector from an image file.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        numpy.ndarray
            Flattened feature vector.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize exactly as done during clustering
        image = cv2.resize(image, self.image_size)

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Flatten into one vector
        feature_vector = image.flatten()

        return feature_vector


# Test
if __name__ == "__main__":

    extractor = FeatureExtractor()

    features = extractor.extract("images/sample.jpg")

    print("Feature length:", len(features))
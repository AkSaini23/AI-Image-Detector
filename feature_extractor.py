
import gldn_encoder
from multi_gldn_histogram import MGLHExtractor
from gldn_encoder import GLDNEncoder
from sobel_gradient_operator import SobelGradientOperator
import numpy as np
class FeatureExtractor:
    def __init__(self, sigmas=[1.0, 1.5, 2.0], num_blocks=(3, 3)):
        self.sigmas = sigmas
        self.mglh_extractor = MGLHExtractor(num_blocks=num_blocks) #Section 3.3

    def extract_features(self, image):
        sobel_operator = SobelGradientOperator(image) #Section 3.1
        Gx, Gy = sobel_operator.apply()
        
        multi_scale_features = []
        
        for sigma in self.sigmas:
            gldn_encoder = GLDNEncoder(Gx, Gy, sigma=sigma) #Section 3.2
            encoded_data = gldn_encoder.execute() # This will generate the compass masks and then compute the multi-channel GLDN codes for the given sigma.
            scale_features = self.mglh_extractor.extract(encoded_data)
    
            multi_scale_features.append(scale_features)
        final_feature_vector = np.concatenate(multi_scale_features)
        
        return final_feature_vector
    
    
    
import cv2

image = cv2.imread('Feature_extractor.jpg', cv2.IMREAD_COLOR)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
extractor = FeatureExtractor(sigmas=[1.0, 1.5, 2.0], num_blocks=(8, 8))
final_features = extractor.extract_features(hsv_image)

print(f"Extraction complete!")
print(np.sum(final_features==0))
print(f"Final MGLH Feature Vector Shape: {final_features.shape}")
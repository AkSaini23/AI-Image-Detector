import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
class SobelGradientOperator:
    def __init__(self, image):
        self.image = image
        self.sobel_x = [[-1, 0, 1],      #Equation 6
                        [-2, 0, 2],
                        [-1, 0, 1]]
        
        self.sobel_y = [[1, 2, 1],        #Equation 7
                        [0, 0, 0],
                        [-1, -2, -1]]

    def apply(self):
        Gx = np.zeros_like(self.image, dtype=np.float32)
        Gy = np.zeros_like(self.image, dtype=np.float32)
        for i in range(3):
            Gx[:,:,i] = ndimage.convolve(self.image[:,:,i], self.sobel_x)   #Equation 8
            Gy[:,:,i] = ndimage.convolve(self.image[:,:,i], self.sobel_y)   #Equation 9
        return Gx, Gy
        
        
""" image = cv2.imread('/Users/akhil.saini042gmail.com/Uni/IT-Security Project/Feature_Extraction/Feature_extractor.jpg', cv2.IMREAD_COLOR)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
sobel_operator = SobelGradientOperator(np.array(hsv_image))
Gx, Gy = sobel_operator.apply()
print("Gradient in x direction:\n", Gx) 
print("Gradient in y direction:\n", Gy)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Gradient in x direction")
plt.imshow(Gx[:,:,0].astype(np.uint8))
plt.subplot(1, 2, 2)
plt.title("Gradient in y direction")
plt.imshow(Gy[:,:,0].astype(np.uint8))
plt.tight_layout()
plt.show() """
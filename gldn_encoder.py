import numpy as np
from scipy.ndimage import convolve, rotate
import matplotlib.pyplot as plt
from sobel_gradient_operator import SobelGradientOperator
import cv2

class GLDNEncoder:
    def __init__(self,Gx, Gy, sigma=1):
        self.Gx = Gx
        self.Gy = Gy
        self.sigma = sigma
        
    def generate_gaussian_mask(self):
        self.kernel_size = int(2 * np.ceil(3 * self.sigma) + 1)
        ax = np.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        x, y = np.meshgrid(ax, ax)
        self.gaussian_mask = (np.exp(-(x**2 + y**2) / (2. * self.sigma**2))  #Equation 11
                              / (2. * np.pi * self.sigma**2)) 
        return self.gaussian_mask, x, y
    
    def generate_compass_masks(self):
        self.compass_masks = []
        self.gaussian_mask, x, y = self.generate_gaussian_mask()
        
        offset = self.kernel_size / 4
        x_shifted = x + offset
    
        self.derivative_gaussian_mask = (                  
            -x_shifted / (self.sigma**2)
            * np.exp(-(x_shifted**2 + y**2) / (2 * self.sigma**2))
            / (2 * np.pi * self.sigma**2)
        )
        mask = convolve(self.derivative_gaussian_mask, self.gaussian_mask)  #Equation 10
        self.compass_masks.append(mask)
        # Generate the remaining 7 masks by rotating the initial mask by multiples of 45 degrees (Compass Masks)
        for i in range(1, 8):
            temp_rotated_mask = rotate(mask, angle=i*45, reshape=False)
            self.compass_masks.append(temp_rotated_mask)
        return self.compass_masks

    def edge_responses(self, gradient_x, gradient_y):
        responses = {
            "x": [],
            "y": []
        }
        for mask in self.compass_masks:
            response_x = convolve(gradient_x, mask) #Equation 15
            response_y = convolve(gradient_y, mask) #Equation 15
            responses["x"].append(response_x)
            responses["y"].append(response_y)
        return np.stack(responses["x"], axis=-1), np.stack(responses["y"], axis=-1)
    
    def encode(self, gradient_x, gradient_y):
        responses_x, responses_y = self.edge_responses(gradient_x, gradient_y)
        x_i_max = np.argmax(responses_x, axis=-1)  #Equation 13
        x_j_min = np.argmin(responses_x, axis=-1)  #Equation 14
        
        y_i_max = np.argmax(responses_y, axis=-1) #Equation 13
        y_j_min = np.argmin(responses_y, axis=-1) #Equation 14
        gldn_x = (x_i_max * 8 + x_j_min).astype(np.uint8)  #Equation 12
        gldn_y = (y_i_max * 8 + y_j_min).astype(np.uint8)  #Equation 12
        return gldn_x, gldn_y
        
        
    def generate_multi_channel_GLDN_code(self):
        encoded_data = {
            "concatenated": [],
            "gldn_x": [],
            "gldn_y": []
        }
        for i in range(self.Gx.shape[2]): # Loop through each color channel (HSV) for both Gx and Gy
            gldn_x, gldn_y = self.encode(self.Gx[:,:,i], self.Gy[:,:,i])
            encoded_data["gldn_x"].append(gldn_x)
            encoded_data["gldn_y"].append(gldn_y)
        encoded_data["gldn_x"] = np.stack(encoded_data["gldn_x"], axis=-1)
        encoded_data["gldn_y"] = np.stack(encoded_data["gldn_y"], axis=-1)
        encoded_data["concatenated"] = np.concatenate([encoded_data["gldn_x"], encoded_data["gldn_y"]], axis=-1)
        return encoded_data
    
    def execute(self):
        self.generate_compass_masks()
        return self.generate_multi_channel_GLDN_code()
    
""" image = cv2.imread('/Users/akhil.saini042gmail.com/Uni/IT-Security Project/Feature_Extraction/Feature_extractor.jpg', cv2.IMREAD_COLOR)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
sobel_operator = SobelGradientOperator(np.array(hsv_image))
Gx, Gy = sobel_operator.apply()

g = GLDNEncoder(Gx, Gy, sigma=1)
M = g.generate_compass_masks()
encoded_data = g.execute()
print("Encoded Data Shape:\n", encoded_data["concatenated"].shape)
print("Encoded Data:\n", encoded_data['gldn_x'].shape)
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(encoded_data['gldn_x'][:,:,i], cmap='gray')
    plt.title(f"GLDN Code Channel {i+1}")
plt.tight_layout()
plt.show()
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(M[i], cmap='gray')
    plt.title(f"Mask {i+1}")
plt.tight_layout()
plt.show() 
print("Total Masks:\n", M) """


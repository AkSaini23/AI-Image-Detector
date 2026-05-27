import numpy as np

class MGLHExtractor:
    def __init__(self, num_blocks=(8, 8) ):
        self.num_blocks_row, self.num_blocks_col = num_blocks
        self.num_bins = 64 # Since GLDN codes are in the range [0, 63], we need 64 bins to cover all possible values.

    def _compute_block_histograms(self, gldn_map):
        row_blocks = np.array_split(gldn_map, self.num_blocks_row, axis=0) # Split the GLDN map into row blocks
        
        histograms = []
        for row_block in row_blocks:
            col_blocks = np.array_split(row_block, self.num_blocks_col, axis=1) # Split each row block into column blocks
            #Now we have blocks of size (H/num_blocks_row, W/num_blocks_col). We compute the histogram for each block.
            
            for block in col_blocks:
                hist, _ = np.histogram(block, bins=self.num_bins, range=(0, self.num_bins))
                histograms.append(hist)
                
        
        return np.concatenate(histograms) #Equation 17 

    def extract(self, encoded_data):
        # encoded_data['concatenated'] contains stacks X and Y gradients 
        # across all 3 color channels, resulting in a shape of (H, W, 6).
        concat_maps = encoded_data["concatenated"]
        num_channels = concat_maps.shape[2]
        
        all_channel_features = []
        
        # Loop through all 3 color channels for X + 3 color channels for Y
        for i in range(num_channels):
            channel_hist = self._compute_block_histograms(concat_maps[:, :, i])
            all_channel_features.append(channel_hist)
            
        return np.concatenate(all_channel_features)
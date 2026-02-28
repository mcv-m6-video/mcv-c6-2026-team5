from sklearn.model_selection import KFold
import numpy as np

class DataSplitter:
    def __init__(self, total_frames):
        self.total_frames = total_frames
        self.indices = np.arange(total_frames)

    def get_split(self, strategy='A', k=4):
        """
        Generator that yields (train_idx, test_idx).
        
        Strategy A: First 25% train, rest test.
        Strategy B: Fixed K-Fold (chunked).
        Strategy C: Random K-Fold (shuffled).
        """
        if strategy == 'A':
            split_point = int(self.total_frames * 0.25)
            # Yields a single tuple (not a loop like K-Fold)
            yield self.indices[:split_point], self.indices[split_point:]
            
        elif strategy == 'B':
            # K-Fold without shuffle (blocks of video)
            kf = KFold(n_splits=k, shuffle=False)
            for train_index, test_index in kf.split(self.indices):
                yield train_index, test_index
                
        elif strategy == 'C':
            # K-Fold WITH shuffle (random frames)
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(self.indices):
                yield train_index, test_index
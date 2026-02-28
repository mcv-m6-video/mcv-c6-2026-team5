from abc import ABC, abstractmethod
import torch

class BaseDetector(ABC):
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model()
        self.model.to(self.device)

    @abstractmethod
    def get_model(self):
        """
        Must return the specific PyTorch model architecture.
        """
        pass

    def predict(self, images, detection_threshold=0.5):
        """
        Standard inference logic for both tasks.
        Args:
            images: List of PIL images or Tensors.
        Returns:
            List of dicts: [{'boxes': [...], 'scores': [...], 'labels': [...]}, ...]
        """
        self.model.eval()
        
        # Convert inputs to tensor if needed and move to device
        # (Assuming inputs are already tensors for simplicity here)
        images = [img.to(self.device) for img in images]

        with torch.no_grad():
            predictions = self.model(images)

        results = []
        for pred in predictions:
            # Filter by score threshold
            keep = pred['scores'] > detection_threshold
            results.append({
                'boxes': pred['boxes'][keep].cpu().numpy(),
                'scores': pred['scores'][keep].cpu().numpy(),
                'labels': pred['labels'][keep].cpu().numpy()
            })
        return results

    def train_step(self, images, targets, optimizer):
        """
        Single training step logic. 
        Override this in off-the-shelf if it shouldn't be trained.
        """
        self.model.train()
        
        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return losses.item()
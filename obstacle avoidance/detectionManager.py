import torch
import torchvision.transforms as T
from PIL import Image
from typing import List
from DTO.detection import Detection
from logger import system_logger


class DetectionManager:
    """Class responsible for object detection using DETR model"""
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.logger = system_logger.detection_logger
        self.setup_model()

    def setup_model(self):
        """Initialize the DETR model with optimizations"""
        try:
            self.transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
            self.model.eval()
            self.logger.info("DETR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def detect(self, image: Image.Image) -> List[Detection]:
        """Perform object detection on the input image"""
        try:
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > self.confidence_threshold
            
            bboxes = self.rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
            
            detections = []
            for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes.tolist()):
                cl = p.argmax()
                confidence = p[cl].item()
                detections.append(Detection(cl, confidence, (xmin, ymin, xmax, ymax)))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            raise

    def rescale_bboxes(self, out_bbox, size):
        """Rescale bounding boxes to image size"""
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def box_cxcywh_to_xyxy(self, x):
        """Convert bounding box format"""
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def is_car_detected(self, detections: List[Detection]) -> bool:
        """Check if a car is detected in the image"""
        for detection in detections:
            if detection.class_id == 3:
                return True
        return False


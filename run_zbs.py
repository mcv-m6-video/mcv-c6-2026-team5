import cv2
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from datetime import datetime  # <-- NUEVO: Para crear carpetas únicas

# --- Project Imports ---
from src.utils.post_processing import get_bboxes_from_mask, merge_bboxes_by_distance
from src.data.parser import load_gt_xml
from src.evaluation.coco_eval import evaluate_coco

# --- Detectron2 / ZBS Imports ---
zbs_path = os.path.abspath("ZBS")
sys.path.insert(0, zbs_path)
sys.path.insert(0, os.path.join(zbs_path, "third_party/CenterNet2"))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from centernet.config import add_centernet_config
from detic.config import add_detic_config

# --- Configuration ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi_2.jpg"
SPLIT_RATIO = 0.25

# =====================================================================
# CLASE ADAPTADORA PARA EL MODELO SOTA (ZBS / Detectron2)
# =====================================================================
class SOTAWrapper:
    def __init__(self, device="cuda"):
        self.device = device
        print("Inicializando modelo SOTA (ZBS/Detectron2)...")
        
        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        
        config_file = "ZBS/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        if os.path.exists(config_file):
            self.cfg.merge_from_file(config_file)
        else:
            print(f"ADVERTENCIA: No se encontró el YAML en {config_file}")
            
        # --- LA MAGIA DEL ZERO-SHOT (SOLUCIÓN AL ERROR) ---
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "ZBS/datasets/metadata/coco_clip_a+cname.npy"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

        if hasattr(self.cfg.MODEL, "CENTERNET"):
            self.cfg.MODEL.CENTERNET.NUM_CLASSES = 80
        # --------------------------------------------------
            
        self.cfg.MODEL.WEIGHTS = "ZBS/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
        
        self.predictor = DefaultPredictor(self.cfg)
        
    def apply(self, frame_bgr):
        outputs = self.predictor(frame_bgr)
        instances = outputs["instances"]
        
        # Clase 2 en COCO = Coche
        cars = instances[instances.pred_classes == 2]
        
        if len(cars) > 0:
            combined_mask = cars.pred_masks.any(dim=0)
        else:
            h, w = frame_bgr.shape[:2]
            combined_mask = torch.zeros((h, w), dtype=torch.bool, device=self.device)
            
        return combined_mask

# =====================================================================

def draw_boxes(frame, boxes, color=(0, 255, 0), label="Car", thickness=2):
    img = frame.copy()
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img

def run_sota_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- NUEVO: Crear carpetas únicas por ejecución ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_sota_{timestamp}")
    masks_dir = os.path.join(run_dir, "masks")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    output_video_path = os.path.join(run_dir, "zbs_sota_experiment.mp4")
    print(f"Resultados de esta ejecución se guardarán en: {run_dir}/")
    # ---------------------------------------------------

    print(f"Abriendo vídeo: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el vídeo en {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    train_len = int(total_frames * SPLIT_RATIO)

    print("Cargando Ground Truth y ROI...")
    gt_boxes = load_gt_xml(GT_PATH)
    gt_boxes_test = {k: v for k, v in gt_boxes.items() if k >= train_len}
    
    roi_mask = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        raise FileNotFoundError(f"ROI mask not found at {ROI_PATH}")
    roi_mask_tensor = torch.from_numpy(roi_mask).to(device).float() / 255.0

    sota_model = SOTAWrapper(device=device)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"Inferencia iniciada. Guardando vídeo en: {output_video_path}")

    pred_boxes_test = {}

    for i in tqdm(range(train_len, total_frames), desc="Procesando Frames con SOTA"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        
        fg_mask = sota_model.apply(frame_bgr)
        
        fg_mask = (fg_mask > 0) & (roi_mask_tensor > 0)
        mask_np = fg_mask.cpu().numpy().astype('uint8') * 255
        
        # --- NUEVO: Guardar la máscara en formato imagen ---
        mask_filename = os.path.join(masks_dir, f"mask_{i:04d}.png")
        cv2.imwrite(mask_filename, mask_np)
        # ---------------------------------------------------
        
        boxes = get_bboxes_from_mask(mask_np, min_area=400) 
        boxes = merge_bboxes_by_distance(boxes, min_distance=31, frame_height=height)
        
        if boxes:
            pred_boxes_test[i] = boxes

        frame_vis = frame_bgr.copy()
        gt_on_frame = gt_boxes_test.get(i, [])
        frame_vis = draw_boxes(frame_vis, gt_on_frame, color=(0, 255, 0), label="GT")
        frame_vis = draw_boxes(frame_vis, boxes, color=(255, 0, 0), label="SOTA")
        cv2.putText(frame_vis, f"Frame: {i} | SOTA Model", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame_vis)

    cap.release()
    writer.release()
    print("Procesamiento de vídeo completado.")

    print("\nCalculando Métricas...")
    map50 = evaluate_coco(gt_boxes_test, pred_boxes_test, height, width)
    
    print("="*40)
    print(f"RESULTADO FINAL SOTA (mAP@0.5): {map50:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_sota_experiment()
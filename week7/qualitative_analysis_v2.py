import pickle
import os
import cv2
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import io # NUEVA IMPORTACIÓN

from model.model_spotting import Model
from util.io import load_json
from util.dataset import load_classes

class DummyArgs:
    def __contains__(self, key):
        return hasattr(self, key)

def load_model_from_config(model_name):
    config_path = f'config/{model_name}.json'
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return None, None
        
    config = load_json(config_path)
    
    m_args = DummyArgs()
    m_args.feature_arch = config.get('feature_arch', 'rny002')
    m_args.num_classes = config.get('num_classes', 12)
    m_args.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    m_args.temporal_head = config.get('temporal_head', 'identity')
    m_args.use_temporal_shift = config.get('use_temporal_shift', False)
    m_args.temporal_shift_fold_div = config.get('temporal_shift_fold_div', 4)
    m_args.tcn_num_layers = config.get('tcn_num_layers', 3)
    m_args.tcn_kernel_size = config.get('tcn_kernel_size', 3)
    m_args.tcn_hidden_dim = config.get('tcn_hidden_dim', None)
    m_args.tcn_dropout = config.get('tcn_dropout', 0.2)
    m_args.tcn_dilations = config.get('tcn_dilations', None)
    m_args.ms_tcn_dilations = config.get('ms_tcn_dilations', [1, 2, 4])
    m_args.ms_tcn_kernel_sizes = config.get('ms_tcn_kernel_sizes', [3, 3, 3])
    m_args.use_actionness = config.get('use_actionness', False)
    m_args.use_temporal_attention = config.get('use_temporal_attention', False)
    m_args.actionness_inference_alpha = config.get('actionness_inference_alpha', 1.0)

    print(f"Initializing {model_name} on {m_args.device}...")
    model = Model(args=m_args)
    
    save_dir = config['save_dir'] + '/' + model_name
    ckpt_path = os.path.join(save_dir, 'checkpoints', 'checkpoint_best.pt')
    
    if os.path.exists(ckpt_path):
        raw_state_dict = torch.load(ckpt_path, map_location=m_args.device)
        clean_state_dict = {k: v for k, v in raw_state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
        model.load(clean_state_dict)
        print(f" -> {model_name} weights loaded successfully.")
    return model, config

def get_top_preds(probs, idx_to_class, top_k=2, threshold=0.15):
    action_probs = probs[1:] 
    top_indices = np.argsort(action_probs)[::-1][:top_k]
    preds = []
    for i in top_indices:
        if action_probs[i] > threshold or len(preds) == 0:
            preds.append(f"{idx_to_class[i + 1]} ({action_probs[i]:.2f})")
    return ", ".join(preds)

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative results for Action Spotting.")
    parser.add_argument('--model_base', type=str, default='baseline')
    parser.add_argument('--model_best', type=str, default='best_model')
    parser.add_argument('--store_dir', type=str, default='/ghome/group05/datasets/WEEK6/SN-BAS-2025_savedata/splits')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--clip_len', type=int, default=50)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--num_clips', type=int, default=5)
    parser.add_argument('--target_class', type=int, default=None)
    parser.add_argument('--indices', type=int, nargs='+', default=None)
    parser.add_argument('--find_improvement', action='store_true')
    
    args = parser.parse_args()

    model_base, _ = load_model_from_config(args.model_base)
    model_best, config_best = load_model_from_config(args.model_best)
    
    classes_dict = load_classes(os.path.join('data', config_best.get('dataset', 'soccernetball'), 'class.txt'))
    idx_to_class = {v: k for k, v in classes_dict.items()}
    ordered_class_names = [idx_to_class[i] for i in range(1, 13)]

    store_path = os.path.join(args.store_dir, f'LEN{args.clip_len}SPLIT{args.split}')
    with open(os.path.join(store_path, 'frame_paths.pkl'), 'rb') as f:
        frame_paths = pickle.load(f)
    with open(os.path.join(store_path, 'labels.pkl'), 'rb') as f:
        labels_store = pickle.load(f)

    if args.indices is not None:
        action_clip_indices = args.indices
    else:
        action_clip_indices = [i for i, labels in enumerate(labels_store) if 0 < len(labels) <= 3]
        if args.target_class:
            action_clip_indices = [i for i in action_clip_indices if any(l['label'] == args.target_class for l in labels_store[i])]

    random.shuffle(action_clip_indices)
    output_dir = os.path.join("qualitative_results", f"SPOTTING_ANIMATED_{args.model_best}")
    os.makedirs(output_dir, exist_ok=True)

    playback_fps = args.fps / args.stride
    generated_count = 0

    # FIJAMOS EL TAMAÑO DE SALIDA PARA QUE EL GRÁFICO SE VEA PERFECTO
    OUT_WIDTH = 800  
    HUD_HEIGHT = 100
    PLOT_HEIGHT = 250

    for idx in action_clip_indices:
        if generated_count >= args.num_clips: break 

        paths_metadata = frame_paths[idx]
        clip_labels = labels_store[idx]
        if not paths_metadata or paths_metadata[1] == -1: continue 
            
        base_path, start, pad_start, pad_end, ndigits, length = paths_metadata
        actual_frame_paths = [None] * pad_start
        for j in range(length - pad_start - pad_end):
            f_name = f"frame{start + j * args.stride}.jpg" if ndigits == -1 else f"{str(start + j * args.stride).zfill(ndigits)}.jpg"
            actual_frame_paths.append(os.path.join(base_path, f_name))
        actual_frame_paths += [None] * pad_end

        tensor_frames = []
        valid_shape = None
        for p in actual_frame_paths:
            if p and os.path.exists(p):
                img = torchvision.io.read_image(p)
                valid_shape = img.shape
                break
        if not valid_shape: continue

        for p in actual_frame_paths:
            tensor_frames.append(torchvision.io.read_image(p) if p and os.path.exists(p) else torch.zeros(valid_shape, dtype=torch.uint8))
        seq = torch.stack(tensor_frames, dim=0)

        with torch.no_grad():
            probs_base_seq = model_base.predict(seq)[0] 
            probs_best_seq = model_best.predict(seq)[0] 
            
        # Filtramos para usar SÓLO las clases del Ground Truth
        gt_classes = set([l['label'] for l in clip_labels])
        
        if args.find_improvement:
            max_probs_base = probs_base_seq[:, 1:].max(axis=0)
            max_probs_best = probs_best_seq[:, 1:].max(axis=0)
            best_better = False
            for gt_cls in gt_classes:
                if max_probs_best[gt_cls - 1] > 0.5 and max_probs_base[gt_cls - 1] < 0.5:
                    best_better = True
                    break
            if not best_better: continue 

        gt_strings = [f"{idx_to_class[l['label']]} (fr {l['label_idx']})" for l in clip_labels]
        gt_string = ", ".join(gt_strings) if gt_strings else "Background"

        print(f"\n[GENERATING] Clip {idx} with synchronized temporal curves...")

        # --- 1. PRE-RENDERIZAR EL GRÁFICO BASE (SIN EL CABEZAL) ---
        fig = plt.figure(figsize=(8, 2.5), dpi=100) # 800x250 pixels
        # Fijamos los márgenes exactos para sincronizar la barra móvil luego
        ax = fig.add_axes([0.1, 0.2, 0.85, 0.7]) 
        
        frames_arr = np.arange(args.clip_len)
        for c_idx in sorted(list(gt_classes)):
            c_name = idx_to_class[c_idx]
            ax.plot(frames_arr, probs_best_seq[:, c_idx], label=f"Best: {c_name}", linewidth=2.5)
            ax.plot(frames_arr, probs_base_seq[:, c_idx], label=f"Base: {c_name}", linewidth=2.5, linestyle='--', alpha=0.5)

        for lbl in clip_labels:
            ax.axvline(x=lbl['label_idx'], color='red', linestyle=':', alpha=0.8)
            ax.text(lbl['label_idx'], 0.95, 'GT', color='red', fontweight='bold', ha='center')

        ax.set_xlim(0, args.clip_len - 1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Confidence")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base_plot_img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), -1)[:, :, :3]
        plt.close(fig)

        # --- 2. GENERAR EL VÍDEO COMPUESTO ---
        first_frame = cv2.imread(next(p for p in actual_frame_paths if p is not None))
        orig_h, orig_w, _ = first_frame.shape
        
        # Calculamos la nueva altura del vídeo al escalarlo al OUT_WIDTH (800)
        scaled_video_h = int(orig_h * (OUT_WIDTH / orig_w))
        total_canvas_h = HUD_HEIGHT + scaled_video_h + PLOT_HEIGHT
        
        out_path = os.path.join(output_dir, f'clip_{idx}_animated.mp4')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (OUT_WIDTH, total_canvas_h))
        
        # Constantes de márgenes de Matplotlib (0.1 izq y 0.85 de ancho -> de 80px a 760px)
        PLOT_LEFT_MARGIN = int(OUT_WIDTH * 0.1)
        PLOT_WIDTH_PX = int(OUT_WIDTH * 0.85)

        for frame_idx, p in enumerate(actual_frame_paths):
            canvas = np.zeros((total_canvas_h, OUT_WIDTH, 3), dtype=np.uint8)
            
            # A. Dibujar Vídeo Redimensionado
            frame = cv2.imread(p) if p and os.path.exists(p) else np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            frame_resized = cv2.resize(frame, (OUT_WIDTH, scaled_video_h))
            canvas[HUD_HEIGHT : HUD_HEIGHT + scaled_video_h, :] = frame_resized
            
            # B. Dibujar HUD (Textos superiores)
            p_base = get_top_preds(probs_base_seq[frame_idx], idx_to_class)
            p_best = get_top_preds(probs_best_seq[frame_idx], idx_to_class)
            cv2.putText(canvas, f"Frame {frame_idx}/{args.clip_len}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"GT:   {gt_string}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, f"Best: {p_best}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(canvas, f"Base: {p_base}", (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            
            # Animación texto de Acción central en el vídeo
            for lbl in clip_labels:
                if abs(lbl['label_idx'] - frame_idx) <= 2:
                    cv2.putText(canvas, f"--> {idx_to_class[lbl['label']]} <--", (OUT_WIDTH//2 - 120, HUD_HEIGHT + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            # C. Dibujar el Gráfico y la línea animada (Cabezal)
            plot_frame = base_plot_img.copy()
            # Calculamos la posición X exacta del cabezal basándonos en los márgenes de Matplotlib
            playhead_x = PLOT_LEFT_MARGIN + int((frame_idx / (args.clip_len - 1)) * PLOT_WIDTH_PX)
            cv2.line(plot_frame, (playhead_x, 0), (playhead_x, PLOT_HEIGHT), (0, 0, 0), 2) # Línea negra vertical
            
            # Pegamos el gráfico al final del canvas
            canvas[HUD_HEIGHT + scaled_video_h :, :] = plot_frame

            out.write(canvas)
            
        out.release()
        generated_count += 1
        
    print("\nGeneration complete!")

if __name__ == '__main__':
    main()
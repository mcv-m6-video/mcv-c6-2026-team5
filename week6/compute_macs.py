import torch
import os
import argparse
from thop import profile, clever_format

# CHANGED: Import from the new spotting model
from model.model_spotting import Model
from util.io import load_json

class DummyArgs:
    """A simple class to mimic the argparse namespace used in the main pipeline."""
    def __contains__(self, key):
        return hasattr(self, key)

def main():
    parser = argparse.ArgumentParser(description="Compute MACs and Params using THOP for Spotting.")
    parser.add_argument('--model', type=str, default='baseline', help='Name of the config file')
    script_args = parser.parse_args()

    # Load configuration
    config_path = f'config/{script_args.model}.json'
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return
        
    config = load_json(config_path)

    # Set up the arguments needed to initialize the model
    args = DummyArgs()
    args.feature_arch = config.get('feature_arch', 'rny002')
    args.num_classes = config.get('num_classes', 12)
    args.clip_len = config.get('clip_len', 50)
    args.device = 'cpu'  # Load on CPU for profiling
    
    # NEW: Spotting Architecture Options
    args.temporal_head = config.get('temporal_head', 'identity')
    args.use_temporal_shift = config.get('use_temporal_shift', False)
    args.temporal_shift_fold_div = config.get('temporal_shift_fold_div', 4)

    # TCN options
    args.tcn_num_layers = config.get('tcn_num_layers', 3)
    args.tcn_kernel_size = config.get('tcn_kernel_size', 3)
    args.tcn_hidden_dim = config.get('tcn_hidden_dim', None)
    args.tcn_dropout = config.get('tcn_dropout', 0.2)
    args.tcn_dilations = config.get('tcn_dilations', None)
    
    args.ms_tcn_dilations = config.get('ms_tcn_dilations', [1, 2, 4])
    args.ms_tcn_kernel_sizes = config.get('ms_tcn_kernel_sizes', [3, 3, 3])

    # Attention & Actionness Options
    args.use_temporal_attention = config.get('use_temporal_attention', False)
    args.use_actionness = config.get('use_actionness', False)

    print(f"Initializing '{script_args.model}' spotting model...")
    model_wrapper = Model(args=args)

    # THOP requires the base PyTorch module, which is wrapped as self._model
    pytorch_module = model_wrapper._model
    pytorch_module.eval() # Ensure it is in evaluation mode

    # Determine dimensions based on your frame_dir resolution (398x224)
    # Shape expected by model: (Batch_size, Clip_Length, Channels, Height, Width)
    b, t, c, h, w = 1, args.clip_len, 3, 224, 398
    dummy_input = torch.randn(b, t, c, h, w)
    
    print(f"\nRunning thop profile with input shape: {dummy_input.shape}...")
    
    # Calculate MACs and parameters
    # Note: We wrap dummy_input in a tuple as expected by THOP
    macs, params = profile(pytorch_module, inputs=(dummy_input, ), verbose=False)
    
    # Format the output for better readability (e.g., converts 1000000 to 1.000M)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
    
    print("\n" + "="*30)
    print("        RESULTS")
    print("="*30)
    print(f"Config File : {script_args.model}.json")
    print(f"Parameters  : {params_formatted}")
    print(f"MACs        : {macs_formatted}")
    print("="*30)

if __name__ == '__main__':
    main()
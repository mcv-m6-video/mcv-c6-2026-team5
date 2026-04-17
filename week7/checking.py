# verify_x3d.py
import torch

x3d = torch.hub.load('facebookresearch/pytorchvideo',
                      'x3d_m', pretrained=True)
x3d.eval()

x = torch.randn(1, 3, 50, 224, 224)

# Ver dimensiones por bloque
print("=== Dimensiones por bloque ===")
feat = x
for i, block in enumerate(x3d.blocks[:-1]):
    feat = block(feat)
    print(f"Block {i}: {feat.shape}")

# Ver VRAM con batch=4
x_batch = torch.randn(4, 3, 50, 224, 224).cuda()
model = torch.nn.Sequential(
    *list(x3d.blocks[:-1])
).cuda()
with torch.no_grad():
    out = model(x_batch)
print(f"\nVRAM usada: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Output shape: {out.shape}")
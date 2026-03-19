import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x):
    return F.normalize(x, p=2, dim=1)


class ViTEmbeddingModel(nn.Module):
    def __init__(self, model_name, embedding_dim, pretrained=False, dropout=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(feat_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

    # def forward(self, x):
    #     feat = self.backbone(x)
    #     feat = self.bn(feat)
    #     feat = self.dropout(feat)
    #     emb = self.embedding(feat)
    #     emb = self.embedding_bn(emb)
    #     return l2_normalize(emb)
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.bn(feat)
        feat = self.dropout(feat)
        emb = self.embedding(feat)
        emb = self.embedding_bn(emb)
        emb_norm = l2_normalize(emb)
        return {
            "embeddings": emb,
            "embeddings_norm": emb_norm,
        }

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("./checkpoints_reid_cityflow/best.pt", map_location=device)
    cfg = ckpt["config"]

    model = ViTEmbeddingModel(
        model_name=cfg["model_name"],
        embedding_dim=cfg["embedding_dim"],
        pretrained=False,
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def extract_embedding(image_path):
        img = Image.open(image_path).convert("RGB")
        x = tfms(img).unsqueeze(0).to(device)
        emb = model(x)[0]
        return emb.cpu()

    emb1 = extract_embedding("crop1.jpg")
    emb2 = extract_embedding("crop2.jpg")
    cosine_sim = torch.dot(emb1, emb2).item()
    print(cosine_sim)
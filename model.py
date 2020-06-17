import torch
import torchvision
import torch.nn.functional as F
from memory_table import MemoryTable
from dataset import Market1501Dataset
import torchvision.transforms as transforms
import utils

class SSLResnet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(SSLResnet, self).__init__()
        base = torchvision.models.resnet50(pretrained=pretrained)
        in_features = list(base.children())[-1].in_features
        self.embedding = torch.nn.Sequential(*list(base.children())[:-2])
        self.dim_reduction = torch.nn.Conv2d(2048, 256, (1,1))
        self.avg_pool = list(base.children())[-2]

    def forward(self, x, embedding=False):
        x = self.embedding(x)
        x = F.relu(x)
        x = self.dim_reduction(x)
        parts = None
        if not embedding:
            parts = x.chunk(8, dim=2)
            parts = list(map(lambda p: F.avg_pool2d(p, kernel_size=(p.shape[2], p.shape[3])), parts))
            parts = list(map(lambda p: p.view(p.shape[0], -1), parts))
            parts = list(map(lambda p: F.normalize(p, p=2), parts))
            parts = torch.stack(parts)
            parts = parts.view(parts.shape[1], parts.shape[0], -1)
        x = F.avg_pool2d(x, kernel_size=(x.size()[2:]))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        return parts, x

    def get_embedding(self, x):
        return self(x, embedding=True)


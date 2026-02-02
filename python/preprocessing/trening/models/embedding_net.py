import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F  # potrzebne do normalizacji L2

class EmbeddingNet(nn.Module):
    """
    Sieć generująca embeddingi twarzy. (np. 128)
    Zamiast klasyfikować obrazy, zamienia je w wektory cech (embeddingi).
    """

    def __init__(self, embedding_size=128, normalize=True):
        """
        Args:
            embedding_size (int): wymiar wyjściowego wektora embeddingu
            normalize (bool): czy normalizować embedding L2
        """
        super().__init__()

        self.normalize = normalize

        backbone = models.resnet18(pretrained=True)

        backbone.fc = nn.Linear(
            backbone.fc.in_features,  # liczba wejściowych cech (512)
            embedding_size            # liczba wyjściowych cech (embedding)
        )

        self.backbone = backbone

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (tensor): batch obrazów twarzy
                        shape: (batch_size, 3, 112, 112)

        Returns:
            tensor: embedding twarzyS
                    shape: (batch_size, embedding_size)
        """
        # 🔹 Przepuszczenie batcha przez ResNet-18
        x = self.backbone(x)

        # 🔹 Normalizacja L2 (każdy embedding ma długość = 1)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x
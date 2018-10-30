import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #Ändra till rtg-nätverk sen.
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune



class CoAttention(nn.Module):

    def __init__(self, encoder_visual_context_dimension, encoder_semantic_context_dimension, hidden_dim, attention_dim):
        super(CoAttention, self).__init__()

        self.encoder_visual_attention = nn.Linear(encoder_visual_context_dimension, attention_dim)  # linear layer to transform encoded visual context
        self.encoder_semantic_attention = nn.Linear(encoder_semantic_context_dimension, attention_dim)  # linear layer to transform encoded semantic context

        self.hidden_dim_attention = nn.Linear(hidden_dim, attention_dim) #hidden dim transform

        self.full_attention = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, visual_features, semantic_features, hidden_lol):
        att1_visual = self.encoder_visual_attention(visual_features)
        att1_semantic = self.encoder_semantic_attention(semantic_features)
        att_hidden = self.hidden_dim_attention(hidden_lol)

        att_fullvisual = self.full_attention(self.tanh(att1_visual + att_hidden.unsqueeze(1))).squeeze(2)
        att_fullsemantic = self.full_attention(self.tanh(att1_semantic + att_hidden.unsqueeze(1))).squeeze(2)

        alpha_visual = self.softmax(att_fullvisual)
        alpha_semantic = self.softmax(att_fullsemantic)


        #etc, fortsätt här







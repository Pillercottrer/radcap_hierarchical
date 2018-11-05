import torch
import numpy as np
import torchvision.transforms as transforms
from skimage.transform import resize
from imageio import imread
from model_hierarchical import Encoder, CoAttention, MLC



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('hello world')


image_path = './000000000139.jpg'

# Read image and process
img = imread(image_path)
if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)
img = resize(img, (256, 256), mode='constant')
img = img.transpose(2, 0, 1)
img = img / 255.
img = torch.FloatTensor(img).to(device)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])
image = transform(img)  # (3, 256, 256)


#Encoding
encoder = Encoder()
encoder = encoder.to(device)

image = image.unsqueeze(0)  # (1, 3, 256, 256)
encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
enc_image_size = encoder_out.size(1)
encoder_dim = encoder_out.size(3)

# Flatten encoding
encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
num_pixels = encoder_out.size(1)

#Multi label classification network
mlc = MLC(num_pixels,encoder_dim, 10)
mlc = mlc.to(device)

mlc.init_weights()
tags = mlc(encoder_out)


#Co-Attention
coAttention = CoAttention(num_pixels, 10, 100, 512, 100)


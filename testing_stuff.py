import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
import nltk
from skimage.transform import resize
from imageio import imread
from model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder
from build_vocab import Vocabulary



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('hello world')


image_path = './000000000139.jpg'
paragraphs = ['A little dog', 'A dog jumping over a fence', 'One small step for mankind']
# Load vocabulary wrapper
with open('./vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

paragraph_tokens =[]

for sent in paragraphs:
    tokens = nltk.tokenize.word_tokenize(str(sent).lower())
    print(tokens)
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    caption = torch.Tensor(caption)
    paragraph_tokens.append(caption)




lengths = [len(sent) for sent in paragraph_tokens]

target_paragraphs = torch.zeros(1, len(paragraph_tokens), max(lengths)).long()

for i, sent_tokens in enumerate(paragraph_tokens):
    target_paragraphs[0, i,:lengths[i]] = sent_tokens

paragraph_sent_lengths = torch.Tensor(lengths)
paragraph_sent_lengths = paragraph_sent_lengths.unsqueeze(0)

num_sents = torch.Tensor([3])
num_sents = num_sents.unsqueeze(0)




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



#Constants
mlc_dim = 25 # num of tags
vocab_size = 11111 # random value for testing
semantic_att_embed_size = 100




#Encoding
encoder = Encoder()
encoder = encoder.to(device)

image = image.unsqueeze(0)  # (1, 3, 256, 256)
encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
enc_image_size = encoder_out.size(1)
encoder_dim = encoder_out.size(3)

#Testing sentence topic generator (håll tummarna)
sent_lstm = SentenceLSTMDecoder(vocab_size)
sent_lstm.to(device)
topic_vectors, stop_vectors = sent_lstm(encoder_out)

#topic_tensor = torch.Tensor(len(topic_vectors), 111)
#topic_tensor = torch.cat(topic_vectors)
#topic_tensor = topic_tensor.unsqueeze(0)

topic_vectors = topic_vectors.unsqueeze(0)

word_lstm = WordLSTMDecoder(vocab_size)
word_lstm.to(device)
word_lstm(topic_vectors, num_sents, target_paragraphs, paragraph_sent_lengths)







# Flatten encoding
encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
num_pixels = encoder_out.size(1)

#Multi label classification network
mlc = MLC(num_pixels,encoder_dim, mlc_dim, vocab_size, semantic_att_embed_size) #visual_attention dim, encoder_dim, tag_dim. vocab_size, embed_dim
mlc = mlc.to(device)

mlc.init_weights()
tags = mlc(encoder_out)

#Embed words
embed = Embedding(vocab_size, semantic_att_embed_size)
embed = embed.to(device)
a_att = embed(tags)


#Co-Attention
h = torch.zeros(512).to(device)
h = h.unsqueeze(0)

#visual_features = encoder_out.contiguous().view(-1, encoder_dim * num_pixels)

coAttention = CoAttention(encoder_dim, semantic_att_embed_size, 100, 512, 100)
coAttention.to(device)
ctx = coAttention(encoder_out, a_att, h)



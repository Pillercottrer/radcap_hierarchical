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

    def __init__(self, encoder_visual_context_dimension, encoder_semantic_context_dimension, ctx_dim, hidden_dim, attention_dim):
        super(CoAttention, self).__init__()

        self.encoder_visual_attention = nn.Linear(encoder_visual_context_dimension, attention_dim)  # linear layer to transform encoded visual context
        self.encoder_semantic_attention = nn.Linear(encoder_semantic_context_dimension, attention_dim)  # linear layer to transform encoded semantic context

        self.hidden_dim_attention_visual = nn.Linear(hidden_dim, attention_dim) #hidden dim transform
        self.hidden_dim_attention_semantic = nn.Linear(hidden_dim, attention_dim)  # hidden dim transform

        self.full_attention_visual = nn.Linear(attention_dim, 1)
        self.full_attention_semantic = nn.Linear(attention_dim, 1)

        self.tanh = nn.Tanh()


        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights


        self.fullyconnected_ctx = nn.Linear(encoder_semantic_context_dimension+encoder_visual_context_dimension, ctx_dim)  #fully connected layer to transform "a_att" and "v_att" to "ctx"

    def forward(self, visual_features, semantic_features, hidden_lol):
        att1_visual = self.encoder_visual_attention(visual_features)
        att1_semantic = self.encoder_semantic_attention(semantic_features)

        att_hidden_visual = self.hidden_dim_attention_visual(hidden_lol)
        att_hidden_semantic = self.hidden_dim_attention_semantic(hidden_lol)

        att_fullvisual = self.full_attention_visual(self.tanh(att1_visual + att_hidden_visual.unsqueeze(1))).squeeze(2)
        att_fullsemantic = self.full_attention_semantic(self.tanh(att1_semantic + att_hidden_semantic.unsqueeze(1))).squeeze(2)

        alpha_visual = self.softmax(att_fullvisual)
        alpha_semantic = self.softmax(att_fullsemantic)

        v_att = (visual_features*alpha_visual.unsqueeze(2)).sum(dim = 1)  #visual attention vector [batch_size, numattentionvecs]
        a_att = (semantic_features*alpha_semantic.unsqueeze(2)).sum(dim = 1)  #semantic attention vector

        concat_att = torch.cat((v_att, a_att), 1) #concatenates v_att and a_att
        ctx = self.fullyconnected_ctx(concat_att)  #co-attention context vector

        return ctx

class MLC(nn.Module):

    def __init__(self, num_pixels, encoder_dimension, tag_dimension, vocab_size, embed_dim):  #from features a single fully connected layer computes tags.
        print('lalal')
        super(MLC, self).__init__()
        self.encoder_dimension = encoder_dimension
        self.tag_dimension = tag_dimension
        self.num_pixels = num_pixels

        self.mlc = nn.Linear(encoder_dimension*num_pixels, tag_dimension)

        self.softmax = nn.Softmax(dim = 1) # 1?

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        # Ändra till rtg-nätverk sen.
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[-2:]
        self.resnet_avgpool = modules[0]
        self.resnet_fc= modules[1]

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.mlc.bias.data.fill_(0)
        self.mlc.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)


    def forward(self, visual_features):
        """Använda 2 sista lagren av resnet istället???"""
        #test = self.resnet_avgpool(visual_features)
        #test = test.squeeze(3)
        #test = test.squeeze(2)

        mlc_encoding = visual_features.contiguous().view(-1, self.encoder_dimension*self.num_pixels)
        tags = self.mlc(mlc_encoding)
        tags_softmax = self.softmax(tags) #????

        sorted, indices = torch.sort(tags_softmax, descending=True)


        return indices[:,:5]



class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.init_weights()

    def forward(self, words):
        return self.embedding(words)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)




class SentenceLSTMDecoder(nn.Module):

    def __init__(self, vocab_size, num_pixels=196, ctx_attention_dim=512, semantic_embed_dim = 512, hidden_dim=512, encoder_visual_dim=2048, ctx_dim = 100, mlc_dim = 15, t_dim = 111, stop_dim=100, max_sents=8, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        #double check.
        super(SentenceLSTMDecoder, self).__init__()

        self.encoder_visual_dim = encoder_visual_dim
        self.ctx_dim = ctx_dim
        self.attention_dim = ctx_attention_dim  #dimension of ctx vector
        self.semantic_embed_dim = semantic_embed_dim  #word embedding?
        self.hidden_dim = hidden_dim  #hidden state dim?
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = CoAttention(encoder_visual_dim, semantic_embed_dim, ctx_dim, hidden_dim, ctx_attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, semantic_embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm_sentence = nn.LSTMCell(ctx_dim, hidden_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(ctx_dim, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(ctx_dim, hidden_dim)  # linear layer to find initial cell state of LSTMCell

        #behövs dessa?
        self.f_beta = nn.Linear(hidden_dim, ctx_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim, vocab_size)  # linear layer to find scores over vocabulary


        #new stuff
        self.max_sentences = max_sents
        self.mlc_dim = mlc_dim
        self.t_dim = t_dim
        self.mlc = MLC(num_pixels, encoder_visual_dim, self.mlc_dim, self.vocab_size, self.semantic_embed_dim)  # visual_attention dim, encoder_dim, tag_dim. vocab_size, embed_dim

        self.fc_h_to_t = nn.Linear(hidden_dim, t_dim)
        self.fc_ctx_to_t = nn.Linear(ctx_dim, t_dim)
        self.tanh = nn.Tanh()

        self.stop_control_1 = nn.Linear(self.hidden_dim, stop_dim) #h(t-1)
        self.stop_control_2 = nn.Linear(self.hidden_dim, stop_dim) # h(t)
        self.stop_control_3 = nn.Linear(stop_dim, 2) #

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        self.init_weights()  # initialize some layers with the uniform distribution



    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

        self.fc_h_to_t.bias.data.fill_(0)
        self.fc_h_to_t.weight.data.uniform_(-0.1, 0.1)
        self.fc_ctx_to_t.bias.data.fill_(0)
        self.fc_ctx_to_t.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)


    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        """
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c

    def forward(self, encoder_out):
        # Multi label classification network
        self.mlc.init_weights()
        tags = self.mlc(encoder_out)

        #set batch-size and encoder_dim
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1) #number of visual feature vectors

        # Multi label classification network
        self.mlc.init_weights()
        tags = self.mlc(encoder_out)

        #semantic attention
        a_att = self.embedding(tags)

        # init_hidden_state
        h, c = self.init_hidden_state(batch_size)  #h_0, c_0
        h_last_time_step = h


        topic_vectors = []
        stop_vectors = []
        for i in range(self.max_sentences):
            # Co-Attention
            ctx = self.attention(encoder_out, a_att, h)  # generate context vector from features and hidden state
            h, c = self.lstm_sentence(ctx, (h, c)) #advance one time-step
            stop = self.softmax(self.tanh(self.stop_control_3(self.stop_control_1(h_last_time_step)+self.stop_control_2(h))))
            stop_vectors.append(stop)
            t = self.tanh(self.fc_h_to_t(h) + self.fc_ctx_to_t(ctx))
            topic_vectors.append(t)
            h_last_time_step = h
            print(stop)

        topic_tensor = torch.Tensor(len(topic_vectors), self.t_dim)
        topic_tensor = torch.cat(topic_vectors)

        stop_tensor = torch.Tensor(len(stop_vectors), 2)
        stop_tensor = torch.cat(stop_vectors)

        return topic_tensor, stop_tensor

class WordLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, topic_vector_size = 111, num_layers = 1, embed_size = 512, hidden_size = 512, max_num_sents = 8, max_seq_length=20):
        super(WordLSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(topic_vector_size, hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.max_num_sents = max_num_sents
        self.vocab_size = vocab_size

        self.init_h = nn.Linear(topic_vector_size, hidden_size)
        self.init_c = nn.Linear(topic_vector_size, hidden_size)

    def init_hidden_state(self, batch_size):
        #h = self.init_h(topic_vectors)
        #c = self.init_c(topic_vectors)
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c

    def forward(self, topic_vectors, num_sents, paragraphs, sentence_lengths):
        """Decode topic vectors and generates captions."""
        """Topic vectors: 3D tensor (batch_size, num_topic vectors, topic_vec_dim)"""
        """Number of sentences is a 1d tensor (batch size, 1)"""
        """Paragraphs is a 3d tensor of  (batch size, max_sents, max_sentence_length)"""
        """Sentence lengths is a 2d tensor (batch size, max_sents)"""

        batch_size = topic_vectors.size(0)
        max_num_topic_vectors = topic_vectors.size(1)
        topic_vector_dim = topic_vectors.size(-1)
        vocab_size = self.vocab_size

        num_sents, sort_ind = num_sents.sort(dim=0, descending=True)
        #sentence_lengths, sort_ind = sentence_lengths.sort(dim=1, descending=True)
        topic_vectors = topic_vectors[sort_ind]
        paragraphs = paragraphs[sort_ind]
        sentence_lengths = sentence_lengths[sort_ind]

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, self.max_num_sents, self.max_seg_length, vocab_size).to(device)

        #naive implementation (consider each topic vector separetely)
        outputs = 'a'
        for i in range(self.max_num_sents):  #Generate sentence from all topic_vectors
            batch_size_i = sum([l > i for l in num_sents])
            topics = topic_vectors[:batch_size_i,i]
            captions = paragraphs[:batch_size_i,i,:]
            lengths = sentence_lengths[:batch_size_i,i]

            # Sort input data by decreasing lengths; why? apparent below (joke, its not so apparent)
            caption_lengths, sort_ind = lengths.sort(dim=0, descending=True)
            topics = topics[sort_ind]
            captions = captions[sort_ind]

            # Embedding
            embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)

            # Initialize LSTM state (0-tensors)
            h, c = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (caption_lengths - 1).tolist()

            # Create tensors to hold word predicion scores and alphas
            #predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

            # Proceed one time-step with topic vector as input
            h, c = self.lstm(topics)
            for j in range(self.max_seg_length):
                #the extra mile
                batch_size_j = sum([l > j for l in num_sents])
                h, c = self.lstm(embeddings[:batch_size_j, j, :], (h[:batch_size_j], c[:batch_size_j]))  # (batch_size_j, decoder_dim)
                preds = self.linear(self.dropout(h))  # (batch_size_j, vocab_size)
                predictions[:batch_size_j, i, j, :] = preds



        return outputs









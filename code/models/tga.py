# from https://github.com/niluthpol/weak_supervised_video_moment

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from modules.tga.model.encoder_image_full import EncoderImageFull
from modules.tga.model.encoder_image_precomp import EncoderImagePrecomp
from modules.tga.model.contrastive_loss import ContrastiveLoss
from modules.tga.utils.l2norm import l2norm
from torch.nn.utils.clip_grad import clip_grad_norm

def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`. We used Precomp
    """
    #print img_dim
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc

# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # tutorials/09 - Image Captioning
        # Build Models
        opt = cfg
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)

    def forward(self, images, captions, lengths, lengths_img):
        # Compute the image and caption embeddings
        # Forward
        cap_init_emb = self.txt_enc(captions, lengths)
        img_emb, attn_weights = self.img_enc(images, cap_init_emb, lengths_img)
        cap_emb = cap_init_emb

        # measure accuracy and record loss
        # self.optimizer.zero_grad()
        loss = self.criterion(img_emb, cap_emb)
        outputs = {"img_emb": img_emb, "cap_emb": cap_emb, "attn_weights": attn_weights}

        return outputs, loss

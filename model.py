"""MSCA model"""
import torch.nn.functional as F
from re import I
import  math
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import os
eps = 5e-11
def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError(
            "Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc
class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
        # self.pool = AP(self.emb_size)

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)

        self.init_weights()
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
    def forward(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)


        # Forward propagate RNN
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] +
                   cap_emb[:, :, cap_emb.size(2)//2:])/2
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len
# RNN GloVe Based Language Model
class GloveEmb(nn.Module):

    def __init__(
            self,
            num_embeddings,
            glove_dim,
            glove_path,
            add_rand_embed=False,
            rand_dim=None,
            **kwargs
        ):
        super(GloveEmb, self).__init__()

        self.num_embeddings = num_embeddings
        self.add_rand_embed = add_rand_embed
        self.glove_dim = glove_dim
        self.final_word_emb = glove_dim

        # word embedding
        self.glove = nn.Embedding(num_embeddings, glove_dim)
        glove = nn.Parameter(torch.load(glove_path))
        self.glove.weight = glove
        self.glove.requires_grad = False

        if add_rand_embed:
            self.embed = nn.Embedding(num_embeddings, rand_dim)
            self.final_word_emb = glove_dim + rand_dim

    def get_word_embed_size(self,):
        return self.final_word_emb

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        emb = self.glove(x)
        if self.add_rand_embed:
            emb2 = self.embed(x)
            emb = torch.cat([emb, emb2], dim=2)

        return emb
class GloveRNNEncoder(nn.Module):

    def __init__(
        self, vocab_size, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        glove_path=None, add_rand_embed=True):

        super(GloveRNNEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        self.embed = GloveEmb(
            vocab_size,
            glove_dim=embed_dim,
            glove_path=glove_path,
            add_rand_embed=add_rand_embed,
            rand_dim=embed_dim,
        )
        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(
            self.embed.final_word_emb,
            latent_size, num_layers,
            batch_first=True,
            bidirectional=use_bi_gru
        )
    def forward(self, captions, lengths):
        emb = self.embed(captions)
        
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] +
                   cap_emb[:, :, cap_emb.size(2)//2:])/2
        # print(cap_emb.shape)
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, cap_len
def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * lamda)
    attn_pos = l1norm(attn_pos, 1)
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)

    return  attn_pos
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w2).clamp(min=eps)).squeeze()
def intra_relation(K, Q, xlambda):
    """
    Q: (n_context, sourceL, d)
    K: (n_context, sourceL, d)
    return (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)
    attn = attn.view(batch_size*KL, KL)
    attn = nn.Softmax(dim=1)(attn*xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn
def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    attn = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, length):

        diagonal = scores.diag().view(length, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self.a_2 = nn.Parameter(torch.ones(size)).cuda()
        self.b_2 = nn.Parameter(torch.zeros(size)).cuda()

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class LearnableProjection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, eps=1e-6):
        super(LearnableProjection, self).__init__()

        # 定义两个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size).cuda()
        self.fc2 = nn.Linear(hidden_size, output_size).cuda()

        # 定义层归一化层
        self.layer_norm = LayerNorm(hidden_size, eps=eps,dim=-1).cuda()
        self.sf=nn.Softmax(dim=1)
    def forward(self, x):
        # 第一个全连接层
        x = self.fc1(x)
        x = self.layer_norm(x)  # 层归一化
        x = torch.relu(x)  # 非线性激活函数

        # 第二个全连接层
        # x = self.fc2(x)
        # x=self.sf(x)
        return x
class MSCA(object):
    def __init__(self, opt):
        # Build Models
        # super(sims_claculator, self).__init__()
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)

        if opt.precomp_enc_text_type == 'basic':
            self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                    opt.embed_size, opt.num_layers,
                                    no_txtnorm=opt.no_txtnorm)
        else:
            if opt.data_name == 'f30k_precomp':
                GloVe_path = opt.vocab_path + 'glove_840B_f30k_precomp.json.pkl'
                self.txt_enc = GloveRNNEncoder(opt.vocab_size, opt.word_dim,
                                        opt.embed_size, opt.num_layers,
                                        use_bi_gru=True, no_txtnorm=False,
                                        glove_path= GloVe_path, add_rand_embed=False)
            else:
                GloVe_path = opt.vocab_path + 'glove_840B_coco_precomp.json.pkl'
                self.txt_enc = GloveRNNEncoder(opt.vocab_size, opt.word_dim,
                                        opt.embed_size, opt.num_layers,
                                        use_bi_gru=True, no_txtnorm=False,
                                        glove_path= GloVe_path, add_rand_embed=False)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
        if opt.data_name == 'coco_precomp':
            opt.learning_rate=0.003
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt, margin=opt.margin)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        self.opt = opt
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        self.sim_tranpar_wv = nn.Linear(1024, 1024).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict
    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    #Calculating Image Anchors and Layering
    def forward_emb(self, images, captions, lengths, volatile=False,*args):
        """

        Calculating Image Anchors and Layering

        """

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        # Forward
        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)

        souryl2 = img_emb.size(1)
        global_img2 = torch.mean(img_emb, dim=1)
        TT2 = global_img2.unsqueeze(1)
        TT_new2 = TT2.repeat(1, 36, 1)
        sim_par_v = torch.pow(torch.sub(img_emb, TT_new2), 2)
        similar2 = l2norm(self.relu(self.sim_tranpar_wv(sim_par_v)), dim=-1)
        similar2 = torch.transpose(similar2, 1, 2)

        # top_k
        K2 = souryl2 // 4  # 设置 K 的值
        values2, indices2 = torch.topk(similar2, k=K2, dim=1, sorted=True)
        attn_new2 = torch.zeros_like(similar2)
        attn_new2.scatter_(1, indices2, 1)

        R_R2 = torch.transpose(attn_new2, 1, 2)
        R_R2 = img_emb * R_R2  # 逐元素相乘
        R_R2 = torch.transpose(R_R2, 1, 2)
        simiv_R2 = torch.bmm(img_emb, R_R2)
        simit_R2 = torch.bmm(cap_emb, R_R2)

        phi2 = LearnableProjection(souryl2, souryl2, souryl2, eps=1e-6)
        simiv_R_new2 = phi2(simiv_R2)
        simit_R_new2 = phi2(simit_R2)

        simiv_R_new2 = torch.transpose(simiv_R_new2, 1, 2)
        simiv_t2 = torch.bmm(simit_R_new2, simiv_R_new2)  # 128 L 36
        simt2=torch.bmm(simiv_t2,img_emb)
        simt2 = 0.05 * l2norm(simt2, -1)
        # simt2=simt2-torch.min(simt2)
        mean = torch.mean(simt2)
        std = torch.std(simt2)
        z_score_matrix = (simt2 - mean) / std
        result1 = torch.where((0 <= (torch.abs(z_score_matrix))) & ((torch.abs(z_score_matrix)) < 3), 1,torch.zeros_like(simt2))
        result2 = torch.where((3 <= (torch.abs(z_score_matrix))) & ((torch.abs(z_score_matrix)) < 4), 1,torch.zeros_like(simt2))
        cap = cap_emb * (1 - result2 - result1)
        cap_emb = cap + (cap_emb * result2 * 3 / 2) + (result1 * simt2 * 0.3) + (cap_emb * result1 * 0.7)

        return img_emb, cap_emb, cap_lens
    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        scores = self.xattn_score(img_emb, cap_emb, cap_len, self.opt)
        loss = self.criterion(scores, img_emb.size(0))

        self.logger.update('Le', loss.item())

        return loss
    def train_emb(self, images, captions, lengths, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths,*args)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
    def xattn_score(self,images, captions, cap_lens, opt):
        similarities = []
        max_pos = []
        max_neg = []
        max_pos_aggre = []
        max_neg_aggre = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        cap_len_i = torch.zeros(1, n_caption)
        n_region = images.size(1)
        batch_size = n_image
        N_POS_WORD = 0
        A = 0
        B = 0
        mean_pos = 0
        mean_neg = 0
        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            cap_len_i[0, i] = n_word
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # text-to-image direction
            contextT = torch.transpose(images, 1, 2)

            attn = torch.bmm(cap_i_expand, contextT)
            attn_i = torch.transpose(attn, 1, 2).contiguous()
            attn_thres = attn - torch.ones_like(attn) * opt.thres

            batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
            attn_row = attn_thres.view(batch_size * queryL, sourceL)
            Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
            attn_neg = Row_max.lt(0).float()
            t2i_sim_neg = Row_max * attn_neg
            # negative effects
            t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)

            # positive attention
            attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.lambda_softmax)
            weiContext_pos = torch.bmm(attn_pos, images)
            t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)

            attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.lambda_softmax)
            t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)

            t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r
            t2i_sim = t2i_sim_neg + t2i_sim_pos

            sim = t2i_sim.mean(dim=1, keepdim=True)
            # # --------------------------------------------------------------------------------------------------------------------------

            wrong_index = sim.sort(0, descending=True)[1][0].item()
            if (wrong_index == i):
                # positive samples
                attn_max_row = torch.max(attn.reshape(batch_size * n_word, n_region).squeeze(), 1)[0].cuda()
                attn_max_row_pos = attn_max_row[(i * n_word): (i * n_word + n_word)].cuda()

                # negative samples
                neg_index = sim.sort(0)[1][0].item()
                attn_max_row_neg = attn_max_row[(neg_index * n_word): (neg_index * n_word + n_word)].cuda()

                max_pos.append(attn_max_row_pos)
                max_neg.append(attn_max_row_neg)
                N_POS_WORD = N_POS_WORD + n_word
                if N_POS_WORD > 200:  # 200 is the empirical value to make adequate samplings
                    max_pos_aggre = torch.cat(max_pos, 0)
                    max_neg_aggre = torch.cat(max_neg, 0)
                    mean_pos = max_pos_aggre.mean().cuda()
                    mean_neg = max_neg_aggre.mean().cuda()
                    stnd_pos = max_pos_aggre.std()
                    stnd_neg = max_neg_aggre.std()

                    A = stnd_pos.pow(2) - stnd_neg.pow(2)
                    B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
                    C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(
                        2) * torch.log(stnd_neg / (opt.alpha * stnd_pos) + 1e-8)

                    thres = opt.thres
                    thres_safe = opt.thres_safe
                    opt.stnd_pos = stnd_pos.item()
                    opt.stnd_neg = stnd_neg.item()
                    opt.mean_pos = mean_pos.item()
                    opt.mean_neg = mean_neg.item()

                    E = B.pow(2) - 4 * A * C
                    if E > 0:
                        #     # A more simple way to calculate the learning boundary after alpha* adjustement
                        #     # In implementation, we can use a more feasible opt.thres_safe, i.e. directly calculate the empirical lower bound, as in the Supplementary Material.
                        #     # (note that alpha* theoretically unifies the opt.thres at training and opt.thres_safe at testing into the same concept)
                        opt.thres = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()
                        opt.thres_safe = (mean_pos - 3 * opt.stnd_pos).item()

                    if opt.thres < 0:
                        opt.thres = 0
                    if opt.thres > 1:
                        opt.thres = 0

                    if opt.thres_safe < 0:
                        opt.thres_safe = 0
                    if opt.thres_safe > 1:
                        opt.thres_safe = 0

                    opt.thres = 0.7 * opt.thres + 0.3 * thres
                    opt.thres_safe = 0.7 * opt.thres_safe + 0.3 * thres_safe
            # # --------------------------------------------------------------------------------------------------------------------------

            if N_POS_WORD < 200:
                opt.thres = 0
                opt.thres_safe = 0

            similarities.append(sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities
    def xattn_score_test(self,images, captions, cap_lens, opt):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        images=images.to(torch.float32)
        captions=captions.to(torch.float32)

        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        n_region = images.size(1)
        batch_size = n_image

        opt.using_intra_info = True

        for i in range(n_caption):
            n_word = cap_lens[i]

            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            contextT = torch.transpose(images, 1, 2)

            # attention matrix between all text words and image regions
            attn = torch.bmm(cap_i_expand, contextT)
            attn_i = torch.transpose(attn, 1, 2).contiguous()
            attn_thres = attn - torch.ones_like(attn) * opt.thres_safe
            # attn_thres = attn - torch.ones_like(attn) * opt.thres

            batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
            attn_row = attn_thres.view(batch_size * queryL, sourceL)
            Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
            if  opt.using_intra_info:
                attn_intra = intra_relation(cap_i, cap_i, 5)
                attn_intra = attn_intra.repeat(batch_size, 1, 1)
                Row_max_intra = torch.bmm(attn_intra, Row_max.reshape(batch_size, n_word).unsqueeze(-1)).reshape(batch_size * n_word, 1)
                attn_neg = Row_max_intra.lt(0).double()
                t2i_sim_neg = Row_max * attn_neg
            else:
                attn_neg = Row_max.lt(0).float()
                t2i_sim_neg = Row_max * attn_neg
            t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)

            attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.lambda_softmax)
            weiContext_pos = torch.bmm(attn_pos, images)
            t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)

            attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.lambda_softmax)
            t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)
            t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r

            t2i_sim =  t2i_sim_neg + t2i_sim_pos
            sim = t2i_sim.mean(dim=1, keepdim=True)
            similarities.append(sim)
        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        return similarities
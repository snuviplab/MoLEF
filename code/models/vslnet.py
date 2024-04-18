# MIT License

# Copyright (c) 2020 ZHANG HAO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from https://github.com/26hzhang/VSLNet

import torch
import torch.nn as nn
from modules.vslnet.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, \
    ConditionedPredictor, HighLightLayer, Conv1D

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        # self.embedding_net = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
        #                                word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
        #                                drop_rate=configs.drop_rate)
        self.embedding_net = Conv1D(in_dim=configs.word_dim, out_dim=configs.dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                             drop_rate=configs.drop_rate)
        self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=4,
                                              max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate)
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        self.predictor = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len, predictor=configs.predictor)
        # init parameters
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)

    def forward(self, words_vec, video_features, video_mask, word_mask, h_labels, start_idx, end_idx) : 
    # def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, h_labels, s_labels, e_labels):
        # v_mask = video_mask, q_mask = word_mask, query_features = words_vec, s_labels = start_idx, e_labels = end_idx, h_labels = localization
        video_features = self.video_affine(video_features)
        query_features = self.embedding_net(words_vec) 
        video_features = self.feature_encoder(video_features, mask=video_mask)
        # query_features = self.feature_encoder(query_features, mask=q_mask)
        query_features = self.feature_encoder(query_features, mask=word_mask)
        features = self.cq_attention(video_features, query_features, video_mask, word_mask)
        features = self.cq_concat(features, query_features, word_mask)
        h_score = self.highlight_layer(features, video_mask)
        features = features * h_score.unsqueeze(2)
        start_logits, end_logits = self.predictor(features, mask=video_mask)
        highlight_loss = self.compute_highlight_loss(h_score, h_labels, video_mask) # h_score
        loc_loss = self.compute_loss(start_logits, end_logits, start_idx, end_idx) # s_labels, e_labels
        total_loss = loc_loss+ 5.0 * highlight_loss
        return start_logits, end_logits, total_loss

    
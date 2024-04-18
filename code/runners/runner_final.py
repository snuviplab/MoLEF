import os, sys, glob
import collections
import logging
from operator import truediv

import numpy as np
import torch.nn as nn
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from runners.build import build_model
from dataloaders.dataloader import get_dataset
from optimizer.adam_optimizer import AdamOptimizer
from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from utils.utils import load_word2vec, AverageMeter, TimeMeter, CountMeter
from utils import io_utils
from utils import metric
from tqdm import tqdm

from modules.tdtan.utils import get_proposal_results

# model output: prediction_boxes, loss 
# for CSMGAN/ReLoCLNet model, use drop_last=True (self.test_loader)

class Runner:
    def __init__(self, args, writer):
        self.writer = writer
        self.num_updates = 0
        self.args = args
        self.word2vec = load_word2vec(args.word2vec_path)
        self._build_loader()
        self._build_model()
        self._build_optimizer()
        self.patience = 0
        self.best_loss = 1000
        self.best_iou = 0

    def _build_loader(self):
        train = get_dataset(self.args.dataset, self.args.model, self.args.feature_path, self.args.train_data,
                            self.word2vec, self.args.max_num_frames, self.args.max_num_words,
                            self.args.max_num_nodes, is_training=True, is_adj=self.args.is_adj, vocab_size=self.args.vocab_size, with_max_IoU=self.args.with_max_IoU)
        val = get_dataset(self.args.dataset, self.args.model, self.args.feature_path, self.args.val_data,
                          self.word2vec, self.args.max_num_frames, self.args.max_num_words,
                          self.args.max_num_nodes, is_training=False, is_adj=self.args.is_adj, vocab_size=self.args.vocab_size, with_max_IoU=self.args.with_max_IoU)
        test = get_dataset(self.args.dataset, self.args.model, self.args.feature_path, self.args.test_data,
                           self.word2vec, self.args.max_num_frames, self.args.max_num_words,
                           self.args.max_num_nodes, is_training=False, is_adj=self.args.is_adj, vocab_size=self.args.vocab_size, with_max_IoU=self.args.with_max_IoU)
        
        if self.args.model == 'tmlga' :
            from modules.tmlga.utils.collate_batch import BatchCollator
            collator = BatchCollator()
            self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=collator)
            self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator) if val else None
            self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=True) if test else None 
        elif self.args.model == 'tga':
            from modules.tga.utils.collate_batch import BatchCollator
            collator = BatchCollator()
            self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=collator)
            self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator) if val else None
            self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator) if test else None 
        elif self.args.model == 'trm':
            from modules.trm.data.collate_batch import BatchCollator
            collator = BatchCollator()
            self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=collator)
            self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator) if val else None
            self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator) if test else None 
        
        else :
            from dataloaders.collate_batch import BatchCollator
            collator = BatchCollator()
            self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=collator)
            self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=False) if val else None
            self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=False) if test else None 
            
            if self.args.model =='csmgan':
                self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=True) if test else None 
            elif self.args.model == 'reloclnet':
                self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=True) if val else None
                self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=True) if test else None 
                

    def _build_model(self):
        self.model = build_model(self.args, self.word2vec)

        print(self.model)
        if self.args.model_load_path:
            self.model.load_state_dict(torch.load(self.args.model_load_path))
            print('Model loaded from ', self.args.model_load_path) 
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1 : 
            self.model = nn.DataParallel(self.model) 
        self.model.to(self.device) 
        
    def _build_optimizer(self):

        if self.args.model == 'drn' and self.args.mode == 'train' :
            learned_params = None
            if self.args.is_first_stage:
                for name, value in self.model.named_parameters():
                    if 'iou_scores' in name or 'mix_fc' in name:
                        value.requires_grad = False
                learned_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
                # self.args.max_num_epochs = 10
            elif self.args.is_second_stage:
                head_params = self.model.module.fcos.head.iou_scores.parameters()
                fc_params = self.model.module.fcos.head.mix_fc.parameters()
                learned_params = list(head_params) + list(fc_params)
                self.args.lr /= 100
            elif self.args.is_third_stage:
                learned_params = list(self.model.parameters())
                self.args.lr /= 10000
            self.optimizer = AdamOptimizer(self.args, learned_params)
            self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)
        
        elif self.args.model == 'reloclnet' :    
            from modules.reloclnet.optimization import BertAdam
            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
            num_train_optimization_steps = len(self.train_loader) * self.args.max_num_epochs
            self.optimizer = BertAdam(optimizer_grouped_parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, warmup=self.args.lr_warmup_proportion,
                                t_total=num_train_optimization_steps, schedule="warmup_linear")
            
        elif self.args.model == 'lvtr':
            from optimizer.lvtr_optimizer import optimizer_scheduler
            backbone_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'backbone' in name and self.args.data_type == 'raw':
                        # param.requires_grad = False
                        backbone_params.append(param)
                    if 'head' in name:
                        head_params.append(param)

            if len(backbone_params) > 0:
                param_dicts = [{'params':backbone_params}, {'params':head_params}]
            else:
                param_dicts = [{'params':head_params}]

            # optimizer and scheduler
            self.optimizer, self.lr_scheduler = optimizer_scheduler(self.args.optimizer, self.args.scheduler, param_dicts, self.args.lr, self.args.wd, self.args.lr_drop_step)
        
        elif self.args.model == 'tga' : 
            from optimizer.lvtr_optimizer import optimizer_scheduler
            # optimizer and scheduler
            self.optimizer, self.lr_scheduler = optimizer_scheduler(self.args.optimizer, self.args.scheduler, list(self.model.parameters()), self.args.lr, self.args.wd, self.args.lr_drop_step)

        elif self.args.model == 'trm' :
            # optimizer and scheduler
            learning_rate = self.args.SOLVER.LR * 1.0
            bert_params = []
            base_params = []
            for name, param in self.model.named_parameters():
                if "bert" in name:
                    bert_params.append(param)
                else:
                    base_params.append(param)
            self.param_dict = {'bert': bert_params, 'base': base_params}
            self.optimizer = optim.AdamW([{'params': base_params},
                                    {'params': bert_params, 'lr': learning_rate * 0.1}], lr=learning_rate, betas=(0.9, 0.99), weight_decay=1e-5)
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.SOLVER.MILESTONES, gamma=0.1)

        elif self.args.model == 'msdetr' :
            from optimizer.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateau
            self.optimizer = AdamOptimizer(self.args, list(self.model.parameters()))
            self.lr_scheduler = ReduceLROnPlateau(self.args, self.optimizer)

        elif self.args.model == 'cnm':
            self.model.froze_mask_generator()
            parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            self.rec_optimizer = AdamOptimizer(self.args, parameters)
            self.rec_lr_scheduler = InverseSquareRootSchedule(self.args, self.rec_optimizer)

            self.model.froze_reconstructor()
            parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            self.mask_optimizer = AdamOptimizer(self.args, parameters)
            self.mask_lr_scheduler = InverseSquareRootSchedule(self.args, self.mask_optimizer)

        else :
            self.optimizer = AdamOptimizer(self.args, list(self.model.parameters()))
            self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)

    def train(self):
        self.best_loss = 1000
        if not os.path.exists(self.args.model_saved_path):
            os.makedirs(self.args.model_saved_path)
        for epoch in range(1, self.args.max_num_epochs + 1):
            logging.info('Start Epoch {}'.format(epoch))
            # for trm
            if self.args.model == 'trm':
                if epoch <= self.args.SOLVER.FREEZE_BERT:
                    for param in self.param_dict['bert']:
                        param.requires_grad_(False)
                else:
                    for param in self.param_dict['bert']:
                        param.requires_grad_(True)
                if epoch <= self.args.SOLVER.ONLY_IOU:
                    logging.info("Using all losses")
                else:
                    logging.info("Using only bce loss")
            self._train_one_epoch(epoch)
            path = os.path.join(self.args.model_saved_path, 'model-%d' % epoch)
            torch.save(self.model.state_dict(), path)
            logging.info('model saved to %s' % path)
            self.eval(epoch)
        logging.info('Done.')

    def _train_one_epoch(self, epoch):
        
        self.model.train()

        epoch_loss_meter = AverageMeter()
        loss_meter = AverageMeter()
        time_meter = TimeMeter()
        max_num_iters = len(self.train_loader)

        # for tmlga model
        # tmlga_vis_train = Visualization(dataset_size=5000)
        
        # for reloclnet
        if self.args.model == 'reloclnet':
            if self.args.hard_negative_start_epoch != -1 and epoch >= self.args.hard_negative_start_epoch:
                self.model.set_hard_negative(True, self.args.hard_pool_size)
            if self.args.train_span_start_epoch != -1 and epoch >= self.args.train_span_start_epoch:
                self.model.set_train_st_ed(self.args.lw_st_ed)
            
            self.use_video = "video" in self.args.ctx_mode
            self.use_sub = "sub" in self.args.ctx_mode
            self.use_tef = "tef" in self.args.ctx_mode
        
        for bid, batch in enumerate(self.train_loader, 1):
            if not self.args.model == 'cnm':
                self.optimizer.zero_grad()

            if self.args.model == 'tmlga' :
                index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
            elif self.args.model == 'tga':
                vid_feats, words_vec, lengths, lengths_img, ids, fr_label, gt, seg, seg_break = batch
            elif self.args.model == 'trm':
                vid_feats, queries, wordlens, all_iou2d, moments, sentences, phrases, duration, index = batch
            else : 
                index, vid_feats, video_mask, words_vec, word_mask, label, \
                scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                sub_feats, sub_mask, tef_feats, words_id, vid_feats_length, words_length, weights, words_feat, localization, start_idx, end_idx, h_labels = batch 
                
            ##################################################################
            if self.args.model in ['ianet', 'cmin', 'csmgan'] :    
                model_input = {
                    'frames': vid_feats.cuda(),
                    'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 'word_mask': word_mask.cuda(),
                    'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'fr_label': label.cuda(),
                    'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                } 

                predict_boxes, loss, _ = self.model(**model_input)

            elif self.args.model in ['tdtan', 'ranet', 'mgpn'] :        
                model_input = {
                    'index': index, 'frames': vid_feats.cuda(), 'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 
                    'word_mask': word_mask.cuda(), 'fr_label': label.cuda(), 'node_pos': id2pos.cuda(), 'map_gt': map_gt.cuda(), 
                    'duration': duration.cuda(), 'gt': gt.cuda()
                }

                joint_prob, loss  = self.model(model_input['words'], model_input['word_mask'], model_input['frames'], model_input['map_gt'])

            elif self.args.model == 'drn' :
                gt_start_end = gt/duration[:,None]
                query_len = torch.from_numpy(np.array([len(words) for words in words_vec]))
                sample_index = np.asarray(sample_index)
                props_list = []
                for i in range(sample_index.shape[1] - 1):
                    props_list.append([sample_index[:,i], sample_index[:,i + 1]])
                props_list.append([sample_index[:,-1], sample_index[:,-1]+1])
                props_start_end = torch.cat([torch.from_numpy((np.array(props_list)/sample_index[:,-1]).astype(np.float32))]).permute([2,0,1])
                
                model_input = {
                    'query_tokens' : words_vec.cuda(), 'query_length' : query_len.cuda(),
                    'props_features' : vid_feats.cuda(),  'props_start_end' : props_start_end.cuda(), 'gt_start_end' : gt_start_end.cuda() 
                }
                box_lists, loss_dict = self.model(**model_input)

                if self.args.is_second_stage :
                    loss = loss_dict['loss_iou'].float()
                else :
                    loss = sum(loss for loss in loss_dict.values())
            
            elif self.args.model == 'tmlga' :
                model_input = {
                    'videoFeat': vid_feats.cuda(), 'videoFeat_lengths': vid_feats_length.cuda(), 'tokens': words_vec.cuda(),
                    'tokens_lengths': words_len.cuda(), 'start': start.cuda(), 'end': end.cuda(), 'localiz': localiz.cuda()
                }
                # loss, indvidual_loss, pred_start, pred_end, attention, attention_loss 
                outputs, loss, loss_dict = self.model(**model_input)

            elif self.args.model == 'reloclnet' :
                from modules.reloclnet.utils import get_match_labels, get_st_ed_label
                
                if self.use_video and self.use_tef :
                    vid_feats = torch.cat([vid_feats, tef_feats], dim=2)
                if self.use_sub and self.use_tef:
                    sub_feats = torch.cat([sub_feats, tef_feats], dim=2)
                    
                
                match_labels = torch.tensor(get_match_labels(label, self.args.max_num_frames))
                model_input = {'query_feat' : words_vec.cuda(), 'query_mask': word_mask.cuda(),
                               'video_feat':vid_feats.cuda(), 'video_mask':video_mask.cuda(),
                               'sub_feat':sub_feats.cuda(), 'sub_mask': sub_mask.cuda(),
                               'st_ed_indices': label.cuda(), 'match_labels': match_labels.cuda()}
                loss, loss_dict = self.model(**model_input)   
            
            elif self.args.model == 'lvtr':
                model_inputs = {'src_txt': words_vec.cuda(non_blocking=self.args.pin_memory),
                                'src_txt_mask': word_mask.cuda(non_blocking=self.args.pin_memory),
                                'src_vid': vid_feats.cuda(non_blocking=self.args.pin_memory),
                                'src_vid_mask': video_mask.cuda(non_blocking=self.args.pin_memory)}

                targets = {}
                xx_spans = label / (self.args.num_input_frames-1)
                center = xx_spans.sum(-1) * 0.5
                width = xx_spans[..., 1] - xx_spans[..., 0]
                stack = torch.stack([center, width], dim=-1)
                targets['target_spans'] = [{"spans": v.unsqueeze(0).cuda(non_blocking=self.args.pin_memory)} for v in stack]

                outputs, loss = self.model(**model_inputs,
                        targets=targets,
                        att_visualize=self.args.att_visualize,
                        corr_visualize=self.args.corr_visualize,
                        epoch_i=epoch,
                        idx=bid)

            elif self.args.model == 'cnm' :
                model_inputs = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(), 
                                'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(),
                                'words_len': words_length.cuda(), 'weights': weights.cuda()}
                self.model.froze_mask_generator()
                self.rec_optimizer.zero_grad()
                outputs, loss, loss_dict = self.model(**model_inputs, loss_type = 'rec')
                loss.backward(retain_graph = True)
                loss_meter.update(loss.item())
                epoch_loss_meter.update(loss.item())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.rec_optimizer.step()
                self.model.froze_reconstructor()
                self.mask_optimizer.zero_grad()
                outputs, loss, loss_dict = self.model(**model_inputs, loss_type = 'ivc', loss_dict = loss_dict)
                loss.backward(retain_graph = True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.mask_optimizer.step()


            elif self.args.model == 'tga' :
                model_input = {'images': vid_feats.cuda(), 'captions': words_vec.cuda(),
                               'lengths': lengths, 'lengths_img': lengths_img}
                outputs, loss = self.model(**model_input)

            elif self.args.model == 'cpl':
                model_inputs = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(),
                                'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(), 'words_len': words_length.cuda(),
                                'weights' : weights.cuda()}
                outputs, loss, loss_dict = self.model(epoch=epoch, **model_inputs)

            elif self.args.model == 'trm':
                model_input = {'feats': vid_feats.cuda(), 'queries': [query.to(self.device) for query in queries], 
                                'wordlens': [torch.tensor(wordlen).to(self.device) if isinstance(wordlen, int) else wordlen.to(self.device) for wordlen in wordlens],
                                'all_iou2d': [iou2d.to(self.device) for iou2d in all_iou2d], 
                               'moments': [moment.to(self.device) for moment in moments], 'sentences': sentences, 'phrases': phrases}
                loss, loss_dict = self.model(**model_input, cur_epoch=epoch)

            elif self.args.model == 'msdetr' :
                msdetr_gt = gt/duration.unsqueeze(1)
                msdetr_gt[:, 1] = torch.where(msdetr_gt[:, 1] > 1, 1, msdetr_gt[:, 1])
                model_input = {'vid_feat' : vid_feats.cuda(), 'txt_feat': words_vec.cuda(), 'gt': msdetr_gt.cuda(), 'word_mask': sub_mask.squeeze(2).cuda(), 
                       'word_label': words_id.cuda(), 'txt_mask': word_mask.cuda(), 'video_id': index}
                losses = self.model(**model_input)
                loss = losses['loss']
<<<<<<< Updated upstream
                
            elif self.args.model in ['lgi', 'plrn'] :
                model_input = {'query_labels': words_vec.cuda(),'query_masks': word_mask.cuda(),
                               'video_feats': vid_feats.cuda(),'video_masks': video_mask.cuda()}
                net_out = self.model(model_input)
                loss_dict = self.model.loss_fn(net_out, gt.cuda(), count_loss=True)
                loss = loss_dict['total_loss']

=======
            
            elif self.args.model == 'vslnet' : 
                model_input = {'words_vec' : words_vec.cuda(), 'video_features': vid_feats.cuda(), 'video_mask': video_mask.cuda(), 'word_mask': word_mask.cuda(), 
                'start_idx': start_idx.cuda(), 'end_idx': end_idx.cuda(), 'h_labels': h_labels.cuda()} 
                s_logits, e_logits, loss = self.model(**model_input)
>>>>>>> Stashed changes
            ##################################################################
            loss = torch.mean(loss)
            
            if self.args.model in ['reloclnet', 'lvtr', 'tga', 'trm'] :
                loss.backward()
                if self.args.model == 'trm':
                    max_norm = 5
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
                curr_lr = float(self.optimizer.param_groups[0]["lr"])
            elif self.args.model == 'cpl':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.num_updates += 1
                curr_lr = self.lr_scheduler.step_update(self.num_updates)
            elif self.args.model == 'cnm':
                self.num_updates += 1
                curr_lr = self.rec_lr_scheduler.step_update(self.num_updates)
                self.mask_lr_scheduler.step_update(self.num_updates)
            else :
                self.optimizer.backward(loss)
                if self.args.model == 'msdetr' :
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm= self.args.clip_grad)
                self.optimizer.step()
                self.num_updates += 1
                curr_lr = self.lr_scheduler.step_update(self.num_updates)

            epoch_loss_meter.update(loss.item())
            loss_meter.update(loss.item())
            time_meter.update()
            
            niter = epoch*max_num_iters + bid
            self.writer.add_scalar('Train/Loss', loss_meter.avg, niter)
            self.writer.add_scalar('Train/Lr', curr_lr, niter)

            if bid % self.args.display_n_batches == 0:
                logging.info('Epoch %d, Batch %d, loss = %.4f, lr = %.5f, %.3f seconds/batch' % (
                    epoch, bid, loss_meter.avg, curr_lr, 1.0 / time_meter.avg
                ))
                loss_meter.reset()
                
        if self.args.model in ['lvtr', 'tga', 'trm']:
            self.lr_scheduler.step()

        self.writer.add_scalar('Train/Epoch_loss', epoch_loss_meter.avg, epoch)

    def eval(self, epoch):
        data_loaders = [self.val_loader, self.test_loader]
        cpu_device = torch.device("cpu")
        meters = collections.defaultdict(lambda: AverageMeter())

        self.model.eval()

        if self.args.model == 'reloclnet':
            self.use_video = "video" in self.args.ctx_mode
            self.use_sub = "sub" in self.args.ctx_mode
            self.use_tef = "tef" in self.args.ctx_mode
        
        if self.args.model in ['lgi', 'plrn']:
            self.model.eval_mode()
            self.model.reset_status()
            self.model.reset_counters()
        
        with torch.no_grad():
            for ind, data_loader in enumerate(data_loaders):
                # for tdtan evaluation
                sorted_segments_list = []
                gt_list = []
                seg_list = []

                # for drn evaluation
                results_dict = {}

                # for tmlga evaluation
                if self.args.model == 'tmlga' :
                    from modules.tmlga.utils.visualization import Visualization
                    tmlga_vis_valid = Visualization(dataset_size=5000, is_train=False)
                    tmlga_vis_test = Visualization(dataset_size=5000, is_train=False)

                # for reloclnet evaluation
                if self.args.model == 'reloclnet':
                    svmr_gt_st_probs = np.zeros((len(data_loader)*self.args.batch_size, self.args.max_num_frames), dtype=np.float32)
                    svmr_gt_ed_probs = np.zeros((len(data_loader)*self.args.batch_size, self.args.max_num_frames), dtype=np.float32)
                
                if self.args.model == 'tga':
                    img_embs = np.zeros((len(data_loader.dataset), self.args.embed_size))
                    cap_embs = np.zeros((len(data_loader.dataset), self.args.embed_size))
                    attention_index = np.zeros((len(data_loader.dataset), 10))
                    rank1_ind = np.zeros((len(data_loader.dataset)))
                    lengths_all = np.zeros((len(data_loader.dataset)))
                    seg_num = np.zeros(len(data_loader.dataset))
                    seg_breaks = np.zeros((len(data_loader.dataset), 3))
                    timestamps = np.zeros((len(data_loader.dataset), 2))
                    gt_labels = np.zeros((len(data_loader.dataset), 2))

                if self.args.model == 'msdetr':
                    from modules.msdetr.kn_util.evaluater import RankMIoUAboveN
                    evaluater1 = RankMIoUAboveN(1, 0.1)
                    evaluater3 = RankMIoUAboveN(1, 0.3)
                    evaluater5 = RankMIoUAboveN(1, 0.5)
                    evaluater7 = RankMIoUAboveN(1, 0.7)

                if self.args.model == 'trm':
                    results_dict = {}
                    duration_dict, moments_dict, sentences_dict = {}, {}, {}
                
                if self.args.model == 'vslnet' : 
                    ious = []

                for bid, batch in enumerate(data_loader, 1):
                    if not self.args.model == 'cnm':
                        self.optimizer.zero_grad()

                    if self.args.model == 'tmlga' :
                        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
                    elif self.args.model == 'tga':
                        vid_feats, words_vec, lengths, lengths_img, ids, fr_label, gt, seg, seg_break = batch
                    elif self.args.model == 'trm':
                        vid_feats, queries, wordlens, all_iou2d, moments, sentences, phrases, duration, index = batch
                    else : 
                        index, vid_feats, video_mask, words_vec, word_mask, label, \
                        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                        sub_feats, sub_mask, tef_feats, words_id, vid_feats_length, words_length, weights, words_feat, localization, start_idx, end_idx, h_labels = batch 
                                      
                    ##################################################################
                    
                    if self.args.model in ['ianet', 'cmin', 'csmgan'] :    
                        model_input = {
                            'frames': vid_feats.cuda(),
                            'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 'word_mask': word_mask.cuda(),
                            'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'fr_label': label.cuda(),
                            'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                        } 

                        predict_boxes, loss, predict_flatten_old = self.model(**model_input)

                    elif self.args.model in ['tdtan', 'ranet', 'mgpn'] :   
                        model_input = {
                            'index': index, 'frames': vid_feats.cuda(), 'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 
                            'word_mask': word_mask.cuda(), 'fr_label': label.cuda(), 'node_pos': id2pos.cuda(), 'map_gt': map_gt.cuda(), 
                            'duration': duration.cuda(), 'gt': gt.cuda()
                        }

                        joint_prob, loss  = self.model(model_input['words'], model_input['word_mask'], model_input['frames'], model_input['map_gt'])
                    
                    elif self.args.model == 'drn' : 
                        gt_start_end = gt/duration[:,None]
                        query_len = torch.from_numpy(np.array([len(words) for words in words_vec]))
                        sample_index = np.asarray(sample_index)
                        props_list = []
                        for i in range(sample_index.shape[1] - 1):
                            props_list.append([sample_index[:,i], sample_index[:,i + 1]])
                        props_list.append([sample_index[:,-1], sample_index[:,-1]+1])
                        props_start_end = torch.cat([torch.from_numpy((np.array(props_list)/sample_index[:,-1]).astype(np.float32))]).permute([2,0,1])
                        
                        model_input = {
                            'query_tokens' : words_vec.cuda(), 'query_length' : query_len.cuda(),
                            'props_features' : vid_feats.cuda(),  'props_start_end' : props_start_end.cuda(), 'gt_start_end' : gt_start_end.cuda() 
                        }

                        box_lists, loss_dict = self.model(**model_input)

                        if self.args.is_second_stage :
                            loss = loss_dict['loss_iou'].float()
                        else :
                            loss = sum(loss for loss in loss_dict.values())

                    elif self.args.model == 'tmlga' :
                        model_input = {
                            'index': index, 
                            'videoFeat': vid_feats.cuda(), 'videoFeat_lengths': vid_feats_length.cuda(), 'tokens': words_vec.cuda(),
                            'tokens_lengths': words_len.cuda(), 'start': start.cuda(), 'end': end.cuda(), 'localiz': localiz.cuda()
                        }
                        # loss, individual_loss, pred_start, pred_end, attention, attention_loss = 
                        outputs, loss, loss_dict = self.model(model_input['videoFeat'], model_input['videoFeat_lengths'], 
                                                                                            model_input['tokens'], model_input['tokens_lengths'], model_input['start'], 
                                                                                            model_input['end'], model_input['localiz'])

                    elif self.args.model == 'reloclnet':
                        from modules.reloclnet.utils import get_match_labels 
                        
                        if self.use_video and self.use_tef :
                            vid_feats = torch.cat([vid_feats, tef_feats], dim=2)
                        if self.use_sub and self.use_tef:
                            sub_feats = torch.cat([sub_feats, tef_feats], dim=2)
                        
                        match_labels = torch.tensor(get_match_labels(label, self.args.max_num_frames))
                        model_input = {'query_feat' : words_vec.cuda(), 'query_mask': word_mask.cuda(),
                               'video_feat':vid_feats.cuda(), 'video_mask':video_mask.cuda(),
                               'sub_feat':sub_feats.cuda(), 'sub_mask': sub_mask.cuda(),
                               'st_ed_indices': label.cuda(), 'match_labels': match_labels.cuda()}
                        loss, loss_dict = self.model(**model_input)     

                    elif self.args.model == 'lvtr' :
                        from modules.lvtr.utils.span_utils import span_cw_to_xx
                        vg_res = []
                        model_inputs = {'src_txt': words_vec.cuda(non_blocking=self.args.pin_memory),
                                        'src_txt_mask': word_mask.cuda(non_blocking=self.args.pin_memory),
                                        'src_vid': vid_feats.cuda(non_blocking=self.args.pin_memory),
                                        'src_vid_mask': video_mask.cuda(non_blocking=self.args.pin_memory)}
                            
                        targets = {}
                        xx_spans = label / (self.args.num_input_frames-1)
                        center = xx_spans.sum(-1) * 0.5
                        width = xx_spans[..., 1] - xx_spans[..., 0]
                        stack = torch.stack([center, width], dim=-1)
                        targets['target_spans'] = [{"spans": v.unsqueeze(0).cuda(non_blocking=self.args.pin_memory)} for v in stack]

                        outputs, loss = self.model(**model_inputs,
                                targets=targets,
                                att_visualize=self.args.att_visualize,
                                corr_visualize=self.args.corr_visualize,
                                epoch_i=epoch,
                                idx=bid)
                        # loss_dict = criterion(outputs, targets)
                        # weight_dict = criterion.weight_dict
                        # loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                        # loss_dict['loss_overall'] = float(loss)

                        timespans = outputs['pred_spans']  # (batch_size, #queries, 2)
                        label_prob = torch.nn.functional.softmax(outputs['pred_logits'], -1)  # (batch_size, #queries, #classes)
                        scores, labels = label_prob.max(-1)  # (batch_size, #queries)

                        for dur, span, score, lbl in zip(duration,
                                                 timespans.cpu(),
                                                 scores.cpu(),
                                                 labels.cpu()):
                            if self.args.span_type == 'cw':
                                spans = torch.clamp(span_cw_to_xx(span), min=0, max=1) * dur

                            # (#queries, 4), [lbl(int), start(float), end(float), score(float)]
                            sorted_preds = torch.cat([lbl[:, None], spans, score[:, None]], dim=1).tolist()
                            if not self.args.no_sort_results:
                                sorted_preds = sorted(sorted_preds, key=lambda x: x[3], reverse=True)

                            sorted_preds = torch.tensor(sorted_preds)
                            sorted_labels = sorted_preds[:, 0].int().tolist()
                            sorted_spans = sorted_preds[:, 1:].tolist()
                            sorted_spans = [[float(f'{e:.4f}') for e in row] for row in sorted_spans]

                            for idx, video_id in enumerate(index):
                                pred_spans = [pred_span for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
                                if len(pred_spans) == 0:
                                    continue
                                cur_query_pred = dict(
                                    video_id=video_id,
                                    query=video_id,
                                    pred_timespan=pred_spans,
                                )
                                vg_res.append(cur_query_pred)

                            ground_truth = []
                            for idx, video_id in enumerate(index):
                                gt_dict = {}
                                gt_dict['video_id'] = video_id
                                gt_dict['query'] = video_id
                                gt_dict['gt_timespan'] = [gt[idx].tolist()]
                                gt_dict['duration'] = duration[idx].tolist()
                                ground_truth.append(gt_dict)

                    elif self.args.model == 'tga':
                        ids = np.array(ids)
                        seg_num[ids] = np.array(seg)
                        seg_breaks[ids] = np.array(seg_break)
                        timestamps[ids] = np.array(fr_label)
                        gt_labels[ids] = np.array(gt)
                        model_input = {'images': vid_feats.cuda(), 'captions': words_vec.cuda(),
                                       'lengths': lengths, 'lengths_img': lengths_img}
                        outputs, loss = self.model(**model_input)

                    elif self.args.model == 'trm':
                        for i, d, m, s in zip(index, duration, moments, sentences):
                            duration_dict[i] = d
                            moments_dict[i] = m
                            sentences_dict[i] = s
                        model_input = {'feats': vid_feats.cuda(), 'queries': [query.to(self.device) for query in queries], 
                               'wordlens': [wordlen.to(self.device) for wordlen in wordlens], 'all_iou2d': [iou2d.to(self.device) for iou2d in all_iou2d], 
                               'moments': [moment.to(self.device) for moment in moments], 'sentences': sentences, 'phrases': phrases}
                        _,_,contrastive_output, iou_output, loss_dict, loss = self.model(**model_input, cur_epoch=epoch)
                    
                    elif self.args.model == 'cpl':
                        model_input = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(),
                                        'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(), 
                                        'words_len': words_length.cuda(), 'weights':weights.cuda()}
                        outputs, loss, loss_dict = self.model(epoch=epoch, **model_input)

                    elif self.args.model == 'msdetr' :
                        msdetr_gt = gt/duration.unsqueeze(1)
                        msdetr_gt[:, 1] = torch.where(msdetr_gt[:, 1] > 1, 1, msdetr_gt[:, 1])
                        model_input = {'vid_feat' : vid_feats.cuda(), 'txt_feat': words_vec.cuda(), 'gt': msdetr_gt.cuda(), 'word_mask': sub_mask.squeeze(2).cuda(), 
                               'word_label': words_id.cuda(), 'txt_mask': word_mask.cuda(), 'video_id': index}
                        losses = self.model(**model_input, mode='test')
                        loss = losses['loss']

                    elif self.args.model == 'cnm' :
                        model_inputs = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(), 
                                        'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(),
                                        'words_len': words_length.cuda(), 'weights': weights.cuda()}
                        outputs = self.model(epoch=epoch, **model_inputs) 
                        
                    elif self.args.model in ['lgi', 'plrn']:
                        model_input = {'query_labels': words_vec.cuda(),'query_masks': word_mask.cuda(),
                                       'video_feats': vid_feats.cuda(),'video_masks': video_mask.cuda()}
                        gts_input = {'timestamps' : gt,
                                     'duration' : duration}
                        net_out = self.model(model_input)
                        loss_dict = self.model.loss_fn(net_out, gt.cuda(), count_loss=True)
                        loss = loss_dict['total_loss']
                        self.model.compute_status(net_out, gts_input)

                    elif self.args.model == 'vslnet': 
                        model_input = {'words_vec' : words_vec.cuda(), 'video_features': vid_feats.cuda(), 'video_mask': video_mask.cuda(), 'word_mask': word_mask.cuda(), 
                        'start_idx': start_idx.cuda(), 'end_idx': end_idx.cuda(), 'h_labels': h_labels.cuda()} 
                        s_logits, e_logits, loss = self.model(**model_input)

                    ##################################################################
                    
                    if not self.args.model == 'cnm':
                        loss = torch.mean(loss)
                        meters['loss'].update(loss.item())
                    
                    if self.args.model in ['ianet', 'cmin', 'csmgan'] :               
                        video_mask = video_mask.cpu().numpy()
                        gt_boxes = model_input['fr_label'].cpu().numpy()
                        predict_boxes = np.round(predict_boxes.cpu().numpy()).astype(np.int32)
                        gt_starts, gt_ends = gt_boxes[:, 0], gt_boxes[:, 1]
                        predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                        predict_starts[predict_starts < 0] = 0
                        seq_len = np.sum(video_mask, -1)
                        predict_ends[predict_ends >= seq_len] = seq_len[predict_ends >= seq_len] - 1
                        IoUs = metric.calculate_IoU_batch((predict_starts, predict_ends),
                                                            (gt_starts, gt_ends))
                        meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])

                        for i in range(1, 10, 2):
                            meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])
                
                    elif self.args.model in ['tdtan', 'ranet', 'mgpn'] :
                        sorted_times = None if self.model.training else get_proposal_results(joint_prob, model_input['duration'])
                        min_idx = min(model_input['index'])
                        batch_indexs = [idx - min_idx for idx in model_input['index']]
                        sorted_segments = [sorted_times[i] for i in batch_indexs]
                        sorted_segments_list.extend(sorted_segments)
                        gt_list.extend(model_input['gt'].cpu().numpy().tolist())
                    
                    elif self.args.model == 'drn' : 
                        for i in range(len(index)) :
                            gt_list = gt_start_end[i].numpy().tolist()
                            per_vid_detections = box_lists[i]["detections"]
                            per_vid_scores = box_lists[i]["scores"]
                            per_vid_level = box_lists[i]['level']
                            props_pred = torch.cat((per_vid_detections, per_vid_scores.unsqueeze(-1)), dim=-1)
                            temp_dict = {'gt': gt_list, 
                                    'node_predictions': props_pred.cpu().numpy().tolist(),
                                    'edge_predictions': props_pred.cpu().numpy().tolist(),
                                    'level': per_vid_level 
                                    } #'query': query,
                            results_dict[index[i]] = []
                            results_dict[index[i]].append(temp_dict)

                    elif self.args.model == 'tmlga':
                        pred_start = outputs['pred_start']
                        pred_end = outputs['pred_end']
                        attention = outputs['attention']
                        individual_loss = loss_dict['individual_loss']
                        attention_loss = loss_dict['atten_loss']
                        if ind == 0 : 
                            tmlga_vis_valid.run(model_input['index'], pred_start, pred_end, model_input['start'], model_input['end'], \
                            model_input['videoFeat_lengths'], epoch, loss.detach(), individual_loss, attention, attention_loss, gt[:, 0], gt[:, 1], factors, fps)
                        elif ind == 1 :
                            tmlga_vis_test.run(model_input['index'], pred_start, pred_end, model_input['start'], model_input['end'], \
                            model_input['videoFeat_lengths'], epoch, loss.detach(), individual_loss, attention, attention_loss, gt[:, 0], gt[:, 1], factors, fps)
                     
                    elif self.args.model == 'reloclnet':
                        
                        
                        _video_feat, _sub_feat = self.model.encode_context(model_input["video_feat"], model_input["video_mask"],
                                                  model_input["sub_feat"], model_input["sub_mask"])
                        # _video_feat, _sub_feat : (B, max_num_frames, frame_dim) 
                        _query_context_scores, _st_probs, _ed_probs = self.model.get_pred_from_raw_query(model_input['query_feat'], model_input['query_mask'], 
                                                           _video_feat, model_input['video_mask'],
                                                           _sub_feat, model_input['sub_mask'], cross=False) # cross에 따라서 shape 바뀜 (Nq, Nv, L)/(N,L)
                        _st_probs = F.softmax(_st_probs, dim=-1) 
                        _ed_probs = F.softmax(_ed_probs, dim=-1)
                        # _st_prob, _ed_prob : (B, max_num_frames)
                    
                        svmr_gt_st_probs[(bid-1) * self.args.batch_size : (bid) * self.args.batch_size, :_st_probs.shape[1]] = \
                            _st_probs.cpu().numpy()
                        svmr_gt_ed_probs[(bid-1) * self.args.batch_size : (bid) * self.args.batch_size, :_ed_probs.shape[1]] = \
                            _ed_probs.cpu().numpy()
                        
                    elif self.args.model == 'lvtr':
                        from modules.lvtr.test import eval_epoch_post_processing
                        self.args.phase = 'val'
                        # path set to /TSGV/results/lvtr/{dataset}/
                        results_filename = f'{self.args.dataset}_' \
                                   f'{self.args.batch_size}b_{self.args.enc_layers}l_{self.args.num_input_frames}f_{self.args.num_proposals}q_' \
                                   f'{self.args.pred_label}_{self.args.set_cost_span}_{self.args.set_cost_giou}_{self.args.set_cost_query}_val.jsonl'
                        metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch_post_processing(
                            self.args, vg_res, ground_truth, results_filename)

                        for iou in self.args.iou :
                            meters['IoU@%.1f' % (iou)].update(metrics_no_nms["brief"]['VG-full-R1@%.1f' % (iou)])
                        meters['mIoU'].update(metrics_no_nms["brief"]["VG-full-mIoU@R1"])

                    elif self.args.model == 'tga':
                        if(outputs['attn_weights'].size(1)<10):
                            attn_weight = torch.zeros(outputs['attn_weights'].size(0), 10, outputs['attn_weights'].size(2))
                            attn_weight[:,0:outputs['attn_weights'].size(1),:] = outputs['attn_weights']
                        else:
                            attn_weight = outputs['attn_weights']

                        batch_length = attn_weight.size(0)
                        attn_weight = torch.squeeze(attn_weight)

                        attn_index = np.zeros((batch_length, 10)) # Rank 1 to 10
                        rank_att1 = np.zeros(batch_length)
                        temp = attn_weight.data.cpu().numpy().copy()
                        for k in range(batch_length):
                            att_weight = temp[k,:]
                            sc_ind = np.argsort(-att_weight)
                            rank_att1[k] = sc_ind[0]
                            attn_index[k,:] = sc_ind[0:10]
                    
                        # preserve the embeddings by copying from gpu and converting to numpy
                        img_embs[ids] = outputs['img_emb'].data.cpu().numpy().copy()
                        cap_embs[ids] = outputs['cap_emb'].data.cpu().numpy().copy()
                        attention_index[ids] = attn_index
                        lengths_all[ids] = lengths_img
                    
                    elif self.args.model == 'msdetr' :
                        scores = losses['scores'].cpu()
                        boxes = losses['boxxes'].cpu()
                        msdetr_gt = msdetr_gt.cpu()
                        # reference_centers = losses['reference_centers'].cpu().numpy()

                        evaluater1.update(boxes, scores, msdetr_gt)
                        evaluater3.update(boxes, scores, msdetr_gt)
                        evaluater5.update(boxes, scores, msdetr_gt)
                        evaluater7.update(boxes, scores, msdetr_gt)

                    elif self.args.model == 'trm':
                        contrastive_output, iou_output = [o.to(cpu_device) for o in contrastive_output], [o.to(cpu_device) for o in iou_output]
                        results_dict.update({video_id: {'contrastive': result1, 'iou': result2} for video_id, result1, result2 in zip(index, contrastive_output, iou_output)})
                            
                    elif self.args.model == 'cnm':
                        bsz = len(duration)
                        gt = gt / duration[:, np.newaxis]
                        width = outputs['width'].view(bsz)
                        center = outputs['center'].view(bsz)
                        selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                                    torch.clamp(center+width/2, max=1)], dim=-1)
                        selected_props = selected_props.cpu().numpy()
                        IoUs = metric.calculate_IoU_batch((selected_props[:, 0], selected_props[:,1]), (gt[:,0], gt[:,1]))
                        meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])
                        for i in range(1, 10, 2):
                            meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])

                    elif self.args.model == 'cpl':
                        num_props = self.model.num_props
                        selected_props = outputs['selected_props'].cpu().numpy()    # (bsz, num_props, 2)
                        gt = gt / duration[:, np.newaxis]
                        bsz = len(duration)

                        # merge selected props (top1)
                        if self.args.strategy == 'vote':
                            if  self.args.dataset == 'Charades':
                                # On charades, the IoU of many proposals is small, and it doesn't make sense to get these proposals to vote. 
                                # So we weight the voting results of each proposal according to it's IoU with the first proposal.
                                c = np.zeros((bsz, num_props))
                                for i in range(num_props):
                                    iou = metric.calculate_IoU_batch((selected_props[:, 0, 0], selected_props[:, 0, 1]), (selected_props[:, i, 0], selected_props[:, i, 1]))
                                    c[:, i] = iou
                            else:
                                c = np.ones((bsz, num_props))
                            votes = np.zeros((bsz, num_props))
                            for i in range(num_props):
                                for j in range(num_props):
                                    iou = metric.calculate_IoU_batch((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                    iou = iou * c[:, j]
                                    votes[:, i] = votes[:, i] + iou
                            idx = np.argmax(votes, axis=1)
                        else:
                            idx = 0
                        result_prop = selected_props[np.arange(bsz), idx] #* duration[:, np.newaxis].cpu().numpy()  # (bsz, 2)
                        IoUs = metric.calculate_IoU_batch((result_prop[:, 0], result_prop[:,1]), (gt[:,0], gt[:,1]))
                        meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])
                        for i in range(1, 10, 2):
                            meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])

                    elif self.args.model == 'vslnet' : 
                        from modules.vslnet.utils.data_util import index_to_time
                        start_indices, end_indices = self.model.extract_index(s_logits, e_logits)
                        start_indices = start_indices.cpu().numpy()
                        end_indices = end_indices.cpu().numpy()
                        duration = duration.cpu().detach().numpy()
                        gt = gt.cpu().detach().numpy()
                        for i, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)) : 
                            start_time, end_time = index_to_time(s_idx, e_idx, self.args.max_num_frames, duration[i])
                            iou = metric.calculate_IoU(i0=[start_time, end_time], i1=gt[i])
                            ious.append(iou)

                if self.args.model == 'tdtan':
                    from modules.tdtan.utils import eval_predictions
                    eval_result, miou = eval_predictions(sorted_segments_list, gt_list, verbose=True)
                
                elif self.args.model == 'ranet' :
                    from modules.ranet.utils import eval_predictions
                    eval_result, miou = eval_predictions(sorted_segments_list, gt_list, verbose=True)

                elif self.args.model == 'mgpn' :
                    from modules.mgpn.utils import eval_predictions
                    eval_result, miou = eval_predictions(sorted_segments_list, gt_list, verbose=True)

                elif self.args.model == 'drn' : 
                    from modules.drn.evaluate_utils import PostProcessRunner
                    iou_topk_dict = {"iou" : self.args.iou, "topk" : self.args.topk}
                    postprocess_runner = PostProcessRunner(results_dict)
                    topks, accuracy_topks, mious = postprocess_runner.run_evaluate(iou_topk_dict=iou_topk_dict, temporal_nms=True)
                    nms_results = postprocess_runner.viz_processed_results
                    
                    for topk in topks:
                        for i in range(len(self.args.iou)):
                            meters['R@%d,IoU@0.%d' % (topk, 2*i+1)].update(accuracy_topks[topk][i]*100, 1)
                            meters['IoU@.%d'%(2*i+1)].update(accuracy_topks[1][i]*100,1) # for tensorboard
                        meters['mIoU'].update(mious[1]*100, 1)
                
                elif self.args.model == 'tmlga' :
                    if ind == 0 : 
                        meters['mIoU'].update(np.mean(tmlga_vis_valid.mIoU))
                        a = tmlga_vis_valid.plot(epoch)
                        for i in range(len(self.args.iou)) :
                            meters['IoU@0.%d' % (2*i+1)].update(a[str(self.args.iou[i])])
                    elif ind == 1 :
                        meters['mIoU'].update(np.mean(tmlga_vis_test.mIoU))
                        a = tmlga_vis_test.plot(epoch)
                        for i in range(len(self.args.iou)) :
                            meters['IoU@0.%d' % (2*i+1)].update(a[str(self.args.iou[i])])
                
                elif self.args.model == 'reloclnet':
                    from modules.reloclnet.inference import get_svmr_res_from_st_ed_probs, eval_by_task_type
                    eval_submission = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, self.args)
                    metrics, metrics_by_type = eval_by_task_type(eval_submission, gt, iou_thds=self.args.iou, recall_topks=(1,5))
                    print("metrics :", metrics)
                    for iou in self.args.iou :
                        meters['IoU@%.1f' % (iou)].update(metrics["{}-r{}".format(iou, 1)])
                        # meters['R@%d, IoU@%.1f' % (1, iou)].update(metrics["{}-r{}".format(iou, 1)])
                        # meters['R@%d, IoU@%.1f' % (5, iou)].update(metrics["{}-r{}".format(iou, 5)])
                    meters['mIoU'].update(metrics["mIoU"])

                elif self.args.model == 'tga':
                    from modules.tga.utils.eval_prediction import t2i
                    eval_result, miou = t2i(img_embs, cap_embs, timestamps, attention_index, lengths_all, 
                                            npts=None, measure=self.args.measure, return_ranks=False, dataset=self.args.dataset, seg_n=seg_num, seg_breaks=seg_breaks)
                    # r11, r13, r15, r17, r1, r5, miou
                    # ious = [r11, r13, r15, r17]
                    for i in range(len(self.args.iou)) :
                         meters['IoU@0.%d' % (2*i+1)].update(eval_result[i][0])
                    # meters['R@1'].update(r1)
                    # meters['R@5'].update(r5)
                    meters['mIoU'].update(miou)

                elif self.args.model == 'trm':
                    from modules.trm.engine.inference import _accumulate_predictions_from_multiple_gpus
                    from modules.trm.data.datasets.evaluation import evaluate
                    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
                    results = evaluate(self.args, dataset=data_loader.dataset, predictions=predictions, nms_thresh=self.args.TEST.NMS_THRESH, duration_l=duration_dict, moments_l=moments_dict, sentences_l=sentences_dict)
                    for i in range(len(self.args.iou)) :
                        meters['IoU@0.%d' % (2*i+1)].update(results['R@1,IoU@0.%d' % (2*i+1)])
                    meters['mIoU'].update(results['mIoU'])

                elif self.args.model == 'msdetr' :
                    meters['IoU@0.1'].update(evaluater1.compute())
                    meters['IoU@0.3'].update(evaluater3.compute())
                    meters['IoU@0.5'].update(evaluater5.compute())
                    meters['IoU@0.7'].update(evaluater7.compute())
                    meters['mIoU'].update(evaluater1.compute_miou())
                
                elif self.args.model in ['lgi', 'plrn']:
                    self.model.save_results("epoch{:03d}".format(epoch), mode="Test")
                    for i in range(len(self.args.iou)):
                        meters['IoU@0.%d' % (2*i+1)].update(self.model.counters['R1-0.%d' % (2*i+1)].get_average())
                    meters['mIoU'].update(self.model.counters['mIoU'].get_average())
                    self.model.reset_counters()

                elif self.args.model == 'vslnet' : 
                    from modules.vslnet.utils.runner_util import calculate_iou_accuracy
                    r1i1 = calculate_iou_accuracy(ious, threshold=0.1)
                    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
                    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
                    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
                    mi = np.mean(ious) * 100.0
                    meters['IoU@0.1'].update(r1i1)
                    meters['IoU@0.3'].update(r1i3)
                    meters['IoU@0.5'].update(r1i5)
                    meters['IoU@0.7'].update(r1i7)
                    meters['mIoU'].update(mi)

                # to wirte tensorboard visulaization
                if ind == 0 :
                    # for validation data
                    print('Validation')
                    self.writer.add_scalar('Val/Loss', meters['loss'].avg, epoch)
                    self.writer.add_scalar('Val/mIoU', meters['mIoU'].avg, epoch)
                    for i in range(len(self.args.iou)) :
                        self.writer.add_scalar('Val/IoU_0.%d'%(2*i+1), meters['IoU@0.%d'%(2*i+1)].avg, epoch)
                else : 
                    # for test data
                    print('Test')
                    self.writer.add_scalar('Test/Loss', meters['loss'].avg, epoch)
                    self.writer.add_scalar('Test/mIoU', meters['mIoU'].avg, epoch)
                    for i in range(len(self.args.iou)) :
                        self.writer.add_scalar('Test/IoU_0.%d'%(2*i+1), meters['IoU@0.%d'%(2*i+1)].avg, epoch)

                # Save the best model to /TSGV/results/lvtr/{dataset}
                if self.args.model in ['lvtr', 'cpl'] and ind == 0:
                    if self.best_loss > meters['loss'].avg:
                        self.patience = 0
                        self.best_loss = meters['loss'].avg
                        if self.args.model == 'lvtr':
                            checkpoint = {
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'lr_scheduler': self.lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'args': self.args
                            }
                            torch.save(
                                checkpoint,
                                os.path.join(
                                    self.args.model_saved_path,
                                    f'best_model_{epoch}_{self.args.dataset}_{self.args.backbone}_' \
                                    f'{self.args.batch_size}b_{self.args.enc_layers}l_{self.args.num_input_frames}f_{self.args.num_proposals}q_' \
                                    f'{self.args.pred_label}_{self.args.set_cost_span}_{self.args.set_cost_giou}_{self.args.set_cost_query}.ckpt'
                                )
                            )
                        else:
                            if not os.path.exists(self.args.train.model_saved_path):
                                os.makedirs(self.args.train.model_saved_path)
                            os.system('cp %s %s'%(os.path.join(self.args.model_saved_path, 'model-%d' % epoch), os.path.join(self.args.train.model_saved_path, 'model-best.pt')))
                            logging.info('Best results have been updated.')
                elif self.args.model == 'cnm' and ind == 0:
                    if self.best_iou < meters['mIoU'].avg :
                        if not os.path.exists(self.args.train.model_saved_path):
                            os.makedirs(self.args.train.model_saved_path)
                        os.system('cp %s %s'%(os.path.join(self.args.model_saved_path, 'model-%d' % epoch), os.path.join(self.args.train.model_saved_path, 'model-best.pt')))
                        logging.info('Best results have been updated.')
                # else:
                #     self.patience += 1

                for key, value in meters.items():
                    print('{}, {:.4f}'.format(key, value.avg), end=' | ')
                    meters[key].reset()
                # if self.patience > self.args.early_stop_patience:
                #     sys.exit(0)
                print()

    def eval_new(self):
        data_loaders = [self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())
        meters_5 = collections.defaultdict(lambda: CountMeter())

        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for bid, batch in enumerate(data_loader, 1):
                    if not self.args.model == 'cnm':
                        self.optimizer.zero_grad()  
                    
                    if self.args.model == 'tmlga' :
                        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
                    elif self.args.model == 'tga' :
                        vid_feats, words_vec, lengths, lengths_img, ids, fr_label, gt, seg, seg_break = batch
                    else : 
                        index, vid_feats, video_mask, words_vec, word_mask, label, \
                        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                        sub_feats, sub_mask, tef_feats, words_id, vid_feats_length, words_length, weights, words_feat = batch 
                    ##################################################################
                    if self.args.model in ['ianet', 'cmin', 'csmgan'] :    
                        model_input = {
                            'frames': vid_feats.cuda(),
                            'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 'word_mask': word_mask.cuda(),
                            'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'fr_label': label.cuda(),
                            'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                        }

                        predict_boxes, loss, predict_flatten_old = self.model(**model_input)

                    elif self.args.model in ['tdtan', 'ranet', 'mgpn'] :   
                        model_input = {
                            'index': index, 'frames': vid_feats.cuda(), 'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 
                            'word_mask': word_mask.cuda(), 'fr_label': label.cuda(), 'node_pos': id2pos.cuda(), 'map_gt': map_gt.cuda(), 
                            'duration': duration.cuda(), 'gt': gt.cuda()
                        }

                        job_prob, loss  = self.model(model_input['words'], model_input['word_mask'], model_input['frames'], model_input['map_gt'])


                    elif self.args.model == 'tga':
                        ids = np.array(ids)
                        seg_num[ids] = np.array(seg)
                        timestamps[ids] = np.array(fr_label)
                        gt_labels[ids] = np.array(gt)
                        model_input = {'images': vid_feats.cuda(), 'captions': words_vec.cuda(),
                                       'lengths': lengths, 'lengths_img': lengths_img}
                        outputs, loss = self.model(**model_input)
                    elif self.args.model == 'cpl':
                        model_input = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(),
                                        'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(), 
                                        'words_len': words_length.cuda(), 'weights':weights.cuda()}
                        outputs, loss, loss_dict = self.model(epoch=0, **model_input)
                    elif self.args.model == 'cnm':
                        model_inputs = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(), 
                                        'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(),
                                        'words_len': words_length.cuda(), 'weights': weights.cuda()}
                        outputs = self.model(epoch=0, **model_inputs) 
                    ##################################################################
                    if not self.args.model == 'cnm':
                        loss = torch.mean(loss)
                        meters['loss'].update(loss.item())

                    if self.args.model == 'cpl':
                        num_props = self.model.num_props
                        selected_props = outputs['selected_props'].cpu().numpy()    # (bsz, num_props, 2)
                        gt = gt / duration[:, np.newaxis]
                        bsz = len(duration)

                        # merge selected props (top1)
                        if self.args.strategy == 'vote':
                            if  self.args.dataset == 'Charades':
                                # On charades, the IoU of many proposals is small, and it doesn't make sense to get these proposals to vote. 
                                # So we weight the voting results of each proposal according to it's IoU with the first proposal.
                                c = np.zeros((bsz, num_props))
                                for i in range(num_props):
                                    iou = metric.calculate_IoU_batch((selected_props[:, 0, 0], selected_props[:, 0, 1]), (selected_props[:, i, 0], selected_props[:, i, 1]))
                                    c[:, i] = iou
                            else:
                                c = np.ones((bsz, num_props))
                            votes = np.zeros((bsz, num_props))
                            for i in range(num_props):
                                for j in range(num_props):
                                    iou = metric.calculate_IoU_batch((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                    iou = iou * c[:, j]
                                    votes[:, i] = votes[:, i] + iou
                            idx = np.argmax(votes, axis=1)
                            top1_res = selected_props[np.arange(bsz), idx] # (bsz, 2)
                        
                        for k in range(selected_props.shape[0]):
                            gt_boxes = gt[k]
                            preds = selected_props[k]
                            if self.args.strategy == 'vote':
                                top1_IoU_matric = metric.compute_IoU_recall([], top1_res[k], gt_boxes, [1])
                                meters_5['mIoU'].update(top1_IoU_matric, 0.5)
                                top5_IoU_matric = metric.compute_IoU_recall([], preds, gt_boxes, [5])
                                meters_5['mIoU'].update(top5_IoU_matric, 0.5)
                            else:
                                topn_IoU_matric = metric.compute_IoU_recall([], preds, gt_boxes)
                                meters_5['mIoU'].update(topn_IoU_matric, 1)
                    
                    elif self.args.model == 'cnm':
                        bsz = len(duration)
                        gt = gt / duration[:, np.newaxis]
                        width = outputs['width'].view(bsz)
                        center = outputs['center'].view(bsz)
                        selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                                    torch.clamp(center+width/2, max=1)], dim=-1)
                        selected_props = selected_props.unsqueeze(1).cpu().numpy()

                        for k in range(selected_props.shape[0]):
                            gt_boxes = gt[k]
                            preds = selected_props[k]
                            topn_IoU_matric = metric.compute_IoU_recall([], preds, gt_boxes)
                            meters_5['mIoU'].update(topn_IoU_matric, 1)
                    
                    else:
                        video_mask_old = video_mask.cpu().numpy()
                        gt_boxes_old = model_input['fr_label'].cpu().numpy()
                        predict_flatten_old = predict_flatten_old.cpu().numpy()
                        predict_boxes_old = np.round(predict_boxes.cpu().numpy()).astype(np.int32)
                        
                        for k in range(predict_boxes.shape[0]):
                            gt_boxes = gt_boxes_old[k]
                            predict_boxes = predict_boxes_old[k]
                            video_mask = video_mask_old[k]
                            predict_flatten = predict_flatten_old[k]
                            gt_starts, gt_ends = gt_boxes[0], gt_boxes[1]
                            predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                            seq_len = np.sum(video_mask, -1)

                            predict_ends[predict_ends >= seq_len] = seq_len -1 
                            predict_starts[predict_starts < 0] = 0

                            predict_boxes[:, 0], predict_boxes[:, 1] = predict_starts, predict_ends
                            
                            topn_IoU_matric = metric.compute_IoU_recall(predict_flatten, predict_boxes, gt_boxes)
                            meters_5['mIoU'].update(topn_IoU_matric, 1)

                print('| ', end='')
                print('---------------')
                IoU_threshs = [0.1, 0.3, 0.5, 0.7]
                top_n_list = [1,5]
                topn_IoU_matric, count = meters_5['mIoU'].val, meters_5['mIoU'].count
                for i in range(2):
                    for j in range(4):
                        print('{}, {:.4f}'.format('IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j]), topn_IoU_matric[i,j]/count), end=' | ')
                meters_5['mIoU'].reset()
                print()

    def eval_save(self):
        data_loaders = [self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())
        meters_5 = collections.defaultdict(lambda: CountMeter())
        time_meter = TimeMeter()
        save_to = os.path.join(self.args.model_saved_path, f"{self.args.model}_{self.args.dataset}_output.pickle")
        # f = open('./our.txt','w')
        
        self.model.eval()

        with torch.no_grad():
            for data_loader in data_loaders:
                # for output pickle 
                output = {}
                output['ids'] = []
                output['predictions'] = []
                output['gts'] = []

                # for tdtan evaluation
                sorted_segments_list = []
                gt_list = []

                # for tmlga evaluation
                if self.args.model == 'tmlga' :
                    from modules.tmlga.utils.visualization import Visualization
                    tmlga_vis_valid = Visualization(dataset_size=5000, is_train=False)
                    tmlga_vis_test = Visualization(dataset_size=5000, is_train=False)
                

                for bid, batch in enumerate(data_loader, 1):
                    if not self.args.model == 'cnm':
                        self.optimizer.zero_grad()
             
                    if self.args.model == 'tmlga' :
                        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
                    elif self.args.model == 'tga':
                        vid_feats, words_vec, lengths, lengths_img, ids, fr_label, gt, seg, seg_break = batch
                    else : 
                        index, vid_feats, video_mask, words_vec, word_mask, label, \
                        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                        sub_feats, sub_mask, tef_feats, words_id, vid_feats_length, words_length, weights, words_feat = batch 
                        
                    ##################################################################

                    if self.args.model in ['ianet', 'cmin', 'csmgan'] :    
                        model_input = {
                            'frames': vid_feats.cuda(),
                            'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 'word_mask': word_mask.cuda(),
                            'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'fr_label': label.cuda(),
                            'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                        } 

                        predict_boxes, loss, predict_flatten_old = self.model(**model_input)

                    elif self.args.model in ['tdtan', 'ranet'] :   
                        model_input = {
                            'index': index, 'frames': vid_feats.cuda(), 'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 
                            'word_mask': word_mask.cuda(), 'fr_label': label.cuda(), 'node_pos': id2pos.cuda(), 'map_gt': map_gt.cuda(), 
                            'duration': duration.cuda(), 'gt': gt.cuda()
                        }

                        joint_prob, loss  = self.model(model_input['words'], model_input['word_mask'], model_input['frames'], model_input['map_gt'])
                    
                    elif self.args.model == 'drn' : 
                        gt_start_end = gt/duration[:,None]
                        query_len = torch.from_numpy(np.array([len(words) for words in words_vec]))
                        sample_index = np.asarray(sample_index)
                        props_list = []
                        for i in range(sample_index.shape[1] - 1):
                            props_list.append([sample_index[:,i], sample_index[:,i + 1]])
                        props_list.append([sample_index[:,-1], sample_index[:,-1]+1])
                        props_start_end = torch.cat([torch.from_numpy((np.array(props_list)/sample_index[:,-1]).astype(np.float32))]).permute([2,0,1])
                        
                        model_input = {
                            'query_tokens' : words_vec.cuda(), 'query_length' : query_len.cuda(),
                            'props_features' : vid_feats.cuda(),  'props_start_end' : props_start_end.cuda(), 'gt_start_end' : gt_start_end.cuda() 
                        }

                        box_lists, loss_dict = self.model(**model_input)

                        if self.args.is_second_stage :
                            loss = loss_dict['loss_iou'].float()
                        else :
                            loss = sum(loss for loss in loss_dict.values())

                    elif self.args.model == 'tmlga' :
                        model_input = {
                            'index': index, 
                            'videoFeat': vid_feats.cuda(), 'videoFeat_lengths': vid_feats_length.cuda(), 'tokens': words_vec.cuda(),
                            'tokens_lengths': words_len.cuda(), 'start': start.cuda(), 'end': end.cuda(), 'localiz': localiz.cuda()
                        }
                        loss, individual_loss, pred_start, pred_end, attention, attention_loss = self.model(model_input['videoFeat'], model_input['videoFeat_lengths'], 
                                                                                            model_input['tokens'], model_input['tokens_lengths'], model_input['start'], 
                                                                                            model_input['end'], model_input['localiz'])

                    elif self.args.model == 'tga':
                        ids = np.array(ids)
                        seg_num[ids] = np.array(seg)
                        timestamps[ids] = np.array(fr_label)
                        gt_labels[ids] = np.array(gt)
                        model_input = {'images': vid_feats.cuda(), 'captions': words_vec.cuda(),
                                       'lengths': lengths, 'lengths_img': lengths_img}
                        outputs, loss = self.model(**model_input)
                    
                    elif self.args.model == 'cpl':
                        model_input = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(),
                                        'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(), 
                                        'words_len': words_length.cuda(), 'weights':weights.cuda()}
                        outputs, loss, loss_dict = self.model(epoch=0, **model_input)

                    elif self.args.model == 'cnm':
                        model_inputs = {'frames_feat': vid_feats.cuda(), 'frames_len': vid_feats_length.cuda(), 
                                        'words_feat': words_feat.cuda(), 'words_id': words_id.cuda(),
                                        'words_len': words_length.cuda(), 'weights': weights.cuda()}
                        outputs = self.model(epoch = 0, **model_inputs) 
                    ##################################################################
                    if not self.args.model == 'cnm':
                        loss = torch.mean(loss)
                        meters['loss'].update(loss.item())
                    time_meter.update()
                    if bid % self.args.display_n_batches == 0:
                        logging.info('%.3f seconds/batch' % (
                            1.0 / time_meter.avg
                        ))

                    

                    if self.args.model in ["ianet", "cmin", "csmgan"] :

                        video_mask_old = video_mask.cpu().numpy()
                        gt_boxes_old = model_input['fr_label'].cpu().numpy()
                        predict_flatten_old = predict_flatten_old.cpu().numpy()
                        predict_boxes_old = np.round(predict_boxes.cpu().numpy()).astype(np.int32)

                        for k in range(predict_boxes.shape[0]):
                            gt_boxes = gt_boxes_old[k]
                            predict_boxes = predict_boxes_old[k]
                            video_mask = video_mask_old[k]
                            predict_flatten = predict_flatten_old[k]
                            gt_starts, gt_ends = gt_boxes[0], gt_boxes[1]
                            predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                            seq_len = np.sum(video_mask, -1)

                            predict_ends[predict_ends >= seq_len] = seq_len -1 
                            predict_starts[predict_starts < 0] = 0
                            predict_boxes[:, 0], predict_boxes[:, 1] = predict_starts, predict_ends
                            
                            topn_IoU_matric = metric.compute_IoU_recall(predict_flatten, predict_boxes, gt_boxes)
                            meters_5['mIoU'].update(topn_IoU_matric, 1)
                            preds = np.concatenate([predict_boxes, np.expand_dims(predict_flatten, axis=1)], axis=1)

                            output['ids'].extend(list(index))
                            output['predictions'].append(preds)
                            output['gts'].append(gt_boxes)

                    elif self.args.model == 'cpl':
                        num_props = self.model.num_props
                        selected_props = outputs['selected_props'].cpu().numpy()    # (bsz, num_props, 2)
                        gt = gt / duration[:, np.newaxis]
                        bsz = len(duration)

                        # merge selected props (top1)
                        if self.args.strategy == 'vote':
                            if  self.args.dataset == 'Charades':
                                # On charades, the IoU of many proposals is small, and it doesn't make sense to get these proposals to vote. 
                                # So we weight the voting results of each proposal according to it's IoU with the first proposal.
                                c = np.zeros((bsz, num_props))
                                for i in range(num_props):
                                    iou = metric.calculate_IoU_batch((selected_props[:, 0, 0], selected_props[:, 0, 1]), (selected_props[:, i, 0], selected_props[:, i, 1]))
                                    c[:, i] = iou
                            else:
                                c = np.ones((bsz, num_props))
                            votes = np.zeros((bsz, num_props))
                            for i in range(num_props):
                                for j in range(num_props):
                                    iou = metric.calculate_IoU_batch((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                    iou = iou * c[:, j]
                                    votes[:, i] = votes[:, i] + iou
                            idx = np.argmax(votes, axis=1)
                            top1_res = selected_props[np.arange(bsz), idx]
                        else:
                            top1_res = selected_props[np.arange(bsz), 0]

                        for k in range(selected_props.shape[0]):
                            gt_boxes = gt[k]
                            preds = selected_props[k] 
                            if self.args.strategy == 'vote':
                                top1_IoU_matric = metric.compute_IoU_recall([], top1_res[k], gt_boxes, [1])
                                meters_5['mIoU'].update(top1_IoU_matric, 0.5)
                                top5_IoU_matric = metric.compute_IoU_recall([], preds, gt_boxes, [5])
                                meters_5['mIoU'].update(top5_IoU_matric, 0.5)
                            else:
                                topn_IoU_matric = metric.compute_IoU_recall([], preds, gt_boxes)
                                meters_5['mIoU'].update(topn_IoU_matric, 1)
                            preds = top1_res[k] * duration[k].cpu().numpy()

                            output['predictions'].append(preds)
                            output['gts'].append(gt_boxes)
                        output['ids'].extend(list(index))

                    elif self.args.model == 'cnm':
                        bsz = len(duration)
                        gt = gt / duration[:, np.newaxis]
                        width = outputs['width'].view(bsz)
                        center = outputs['center'].view(bsz)
                        selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                                    torch.clamp(center+width/2, max=1)], dim=-1)
                        selected_props = selected_props.unsqueeze(1).cpu().numpy()

                        for k in range(selected_props.shape[0]):
                            gt_boxes = gt[k]
                            preds = selected_props[k]
                            topn_IoU_matric = metric.compute_IoU_recall([], preds, gt_boxes)
                            meters_5['mIoU'].update(topn_IoU_matric, 1)
                            preds = selected_props[k] * duration[k].cpu().numpy()

                            output['predictions'].append(preds)
                            output['gts'].append(gt_boxes)
                        output['ids'].extend(list(index))

                io_utils.write_pkl(save_to, output) # save! 

                IoU_threshs = [0.1, 0.3, 0.5, 0.7]
                top_n_list = [1,5]
                topn_IoU_matric, count = meters_5['mIoU'].val, meters_5['mIoU'].count
                for i in range(2):
                    for j in range(4):
                        print('{}, {:.4f}'.format('IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j]), topn_IoU_matric[i,j]/count), end=' | ')
                meters_5['mIoU'].reset()

                print()

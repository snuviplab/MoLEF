import collections
import logging
from operator import truediv
import os

import torch
from torch.utils.data import DataLoader
from utils import metric
from dataloaders.dataloader import get_dataset
from optimizer.adam_optimizer import AdamOptimizer
from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from utils.utils import load_word2vec, AverageMeter, TimeMeter, CountMeter
from utils import io_utils
import numpy as np
import torch.nn as nn
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

    def _build_loader(self):
        train = get_dataset(self.args.dataset, self.args.model, self.args.feature_path, self.args.train_data,
                            self.word2vec, self.args.max_num_frames, self.args.max_num_words,
                            self.args.max_num_nodes, is_training=True, is_adj=self.args.is_adj, with_max_IoU=self.args.with_max_IoU)
        val = get_dataset(self.args.dataset, self.args.model, self.args.feature_path, self.args.val_data,
                          self.word2vec, self.args.max_num_frames, self.args.max_num_words,
                          self.args.max_num_nodes, is_training=False, is_adj=self.args.is_adj, with_max_IoU=self.args.with_max_IoU)
        test = get_dataset(self.args.dataset, self.args.model, self.args.feature_path, self.args.test_data,
                           self.word2vec, self.args.max_num_frames, self.args.max_num_words,
                           self.args.max_num_nodes, is_training=False, is_adj=self.args.is_adj, with_max_IoU=self.args.with_max_IoU)
        
        if self.args.model == 'tmlga' :
            from modules.tmlga.utils.collate_batch import BatchCollator
            collator = BatchCollator()
            self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=collator)
            self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator) if val else None
            self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=True) if test else None 
        else :
            from dataloaders.collate_batch import BatchCollator
            collator = BatchCollator()
            self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=collator)
            self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=False) if val else None
            self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                        shuffle=False, collate_fn=collator, drop_last=False) if test else None 

    def _build_model(self):
        if self.args.model == 'cmin' :
            from models.cmin import Model 
            self.model = Model(self.args)

        elif self.args.model == 'csmgan' :
            from models.csmgan import Model
            self.model = Model(self.args)

        elif self.args.model == 'tdtan' :
            from models.tdtan import Model
            self.model = Model()
            
        elif self.args.model == 'ranet' :
            from models.ranet import Model
            self.model = Model()

        elif self.args.model == 'ianet' :
            from models.ianet import Model
            self.model = Model(self.args)
        
        elif self.args.model == 'drn' :
            from models.drn import Model
            self.model = Model(len(self.word2vec.index_to_key), self.args)
            
        elif self.args.model == 'tmlga' :
            from models.tmlga import Model
            from modules.tmlga.config import cfg
            self.model = Model(cfg)
            
        elif self.args.model == 'reloclnet':
            from models.reloclnet import Model
            self.model = Model(self.args)

        print(self.model)
        if self.args.model_load_path:
            self.model.load_state_dict(torch.load(self.args.model_load_path))
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1 : 
            self.model = nn.DataParallel(self.model) 
        self.model.to(device) 
        
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
        else :
            self.optimizer = AdamOptimizer(self.args, list(self.model.parameters()))
            self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)

    def train(self):
        if not os.path.exists(self.args.model_saved_path):
            os.makedirs(self.args.model_saved_path)
        for epoch in range(1, self.args.max_num_epochs + 1):
            logging.info('Start Epoch {}'.format(epoch))
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
            self.optimizer.zero_grad()

            if self.args.model == 'tmlga' :
                index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
            else : 
                index, vid_feats, video_mask, words_vec, word_mask, label, \
                scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                sub_feats, sub_mask, tef_feats = batch 
                
            ##################################################################
            if self.args.model in ['ianet', 'cmin', 'csmgan'] :    
                model_input = {
                    'frames': vid_feats.cuda(),
                    'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 'word_mask': word_mask.cuda(),
                    'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'fr_label': label.cuda(),
                    'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                } 

                predict_boxes, loss, _ = self.model(**model_input)

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
                    'videoFeat': vid_feats.cuda(), 'videoFeat_lengths': vid_feats_length.cuda(), 'tokens': words_vec.cuda(),
                    'tokens_lengths': words_len.cuda(), 'start': start.cuda(), 'end': end.cuda(), 'localiz': localiz.cuda()
                }
                loss, indvidual_loss, pred_start, pred_end, attention, attention_loss = self.model(**model_input)

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
            
            ##################################################################

            loss = torch.mean(loss)
            
            if self.args.model == 'reloclnet' :
                loss.backward()
                self.optimizer.step()
                curr_lr = float(self.optimizer.param_groups[0]["lr"])
            
            else :
                self.optimizer.backward(loss)
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

        self.writer.add_scalar('Train/Epoch_loss', epoch_loss_meter.avg, epoch)

    def eval(self, epoch):
        data_loaders = [self.val_loader, self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())

        self.model.eval()
        
        if self.args.model == 'reloclnet':
            self.use_video = "video" in self.args.ctx_mode
            self.use_sub = "sub" in self.args.ctx_mode
            self.use_tef = "tef" in self.args.ctx_mode
        
        with torch.no_grad():
            for ind, data_loader in enumerate(data_loaders):
                # for tdtan evaluation
                sorted_segments_list = []
                gt_list = []

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
                
                for bid, batch in enumerate(data_loader, 1):
                    self.optimizer.zero_grad()

                    if self.args.model == 'tmlga' :
                        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
                    else : 
                        index, vid_feats, video_mask, words_vec, word_mask, label, \
                        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                        sub_feats, sub_mask, tef_feats = batch 
                        
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

                    ##################################################################

                    loss = torch.mean(loss)

                    meters['loss'].update(loss.item())
                    
                    if self.args.model in ['ianet', 'cmin', 'csmgan'] : 
                            
                        video_mask = video_mask.cpu().numpy()
                        gt_boxes = model_input['fr_label'].cpu().numpy()
                        predict_boxes = np.round(predict_boxes.cpu().numpy()).astype(np.int32)
                        print(predict_boxes.shape)
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
                
                    elif self.args.model in ['tdtan', 'ranet'] :
                        # print("joint prob", joint_prob.shape) # (b, 1, 64, 64)
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
                            # try:
                            #     results_dict[index[i]].append(temp_dict)
                            # except KeyError:
                            results_dict[index[i]] = []
                            results_dict[index[i]].append(temp_dict)

                    elif self.args.model == 'tmlga':
                        if ind == 0 : 
                            tmlga_vis_valid.run(model_input['index'], pred_start, pred_end, model_input['start'], model_input['end'], \
                            model_input['videoFeat_lengths'], epoch, loss.detach(), individual_loss, attention, attention_loss, gt[:, 0], gt[:, 1], factors, fps)
                        elif ind == 1 :
                            tmlga_vis_test.run(model_input['index'], pred_start, pred_end, model_input['start'], model_input['end'], \
                            model_input['videoFeat_lengths'], epoch, loss.detach(), individual_loss, attention, attention_loss, gt[:, 0], gt[:, 1], factors, fps)
                     
                    elif self.args.model == 'reloclnet':
                        import torch.nn.functional as F
                        
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
                            
                if self.args.model == 'tdtan':
                    from modules.tdtan.utils import eval_predictions
                    # evaluation for all data
                    # with open("tdtan_output%s.txt".format(self.args.dataset), 'w', encoding='UTF-8') as f:
                    #     for segment in sorted_segments_list :
                    #         f.write(segment+"\n")
                    # with open("tdtan_gt%s.txt".format(self.args.dataset), 'w', encoding='UTF-8') as f:
                    #     for segment in gt_list : 
                    #         f.write(segment+"\n")
                    eval_result, miou = eval_predictions(sorted_segments_list, gt_list, verbose=True)
                
                elif self.args.model == 'ranet' :
                    from modules.ranet.utils import eval_predictions
                    eval_result, miou = eval_predictions(sorted_segments_list, gt_list, verbose=True)
                
                elif self.args.model == 'drn' : 
                    from modules.drn.evaluate_utils import PostProcessRunner
                    iou_topk_dict = {"iou" : self.args.iou, "topk" : self.args.topk}
                    postprocess_runner = PostProcessRunner(results_dict)
                    topks, accuracy_topks, mious = postprocess_runner.run_evaluate(iou_topk_dict=iou_topk_dict, temporal_nms=True)
                    nms_results = postprocess_runner.viz_processed_results
                    # if ind == 1 : # test prediction
                    #     import json
                    #     with open(f"drn_{self.args.dataset.lower()}_predictions.json", "w") as fp :
                    #         json.dump(nms_results, fp, indent='\t')
                            
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
                    
                # to wirte tensorboard visulaization
                if ind == 0 :
                    # for validation data
                    self.writer.add_scalar('Val/Loss', meters['loss'].avg, epoch)
                    self.writer.add_scalar('Val/mIoU', meters['mIoU'].avg, epoch)
                    for i in range(len(self.args.iou)) :
                        self.writer.add_scalar('Val/IoU_0.%d'%(2*i+1), meters['IoU@0.%d'%(2*i+1)].avg, epoch)
                else : 
                    # for test data
                    self.writer.add_scalar('Test/Loss', meters['loss'].avg, epoch)
                    self.writer.add_scalar('Test/mIoU', meters['mIoU'].avg, epoch)
                    for i in range(len(self.args.iou)) :
                        self.writer.add_scalar('Test/IoU_0.%d'%(2*i+1), meters['IoU@0.%d'%(2*i+1)].avg, epoch)

                for key, value in meters.items():
                    print('{}, {:.4f}'.format(key, value.avg), end=' | ')
                    meters[key].reset()
                print()

    def eval_new(self):
        data_loaders = [self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())
        meters_5 = collections.defaultdict(lambda: CountMeter())

        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for bid, batch in enumerate(data_loader, 1):
                    self.optimizer.zero_grad()
                    
                    if self.args.model == 'tmlga' :
                        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
                    else : 
                        index, vid_feats, video_mask, words_vec, word_mask, label, \
                        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                        sub_feats, sub_mask, tef_feats = batch 
                        
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

                        job_prob, loss  = self.model(model_input['words'], model_input['word_mask'], model_input['frames'], model_input['map_gt'])

                    ##################################################################

                    loss = torch.mean(loss)

                    meters['loss'].update(loss.item())
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
                    self.optimizer.zero_grad()
             
                    if self.args.model == 'tmlga' :
                        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
                    else : 
                        index, vid_feats, video_mask, words_vec, word_mask, label, \
                        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
                        sub_feats, sub_mask, tef_feats = batch 
                        
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

                    ##################################################################

                    loss = torch.mean(loss)

                    time_meter.update()
                    if bid % self.args.display_n_batches == 0:
                        logging.info('%.3f seconds/batch' % (
                            1.0 / time_meter.avg
                        ))

                    meters['loss'].update(loss.item())

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

                io_utils.write_pkl(save_to, output) # save! 

                IoU_threshs = [0.1, 0.3, 0.5, 0.7]
                top_n_list = [1,5]
                topn_IoU_matric, count = meters_5['mIoU'].val, meters_5['mIoU'].count
                for i in range(2):
                    for j in range(4):
                        print('{}, {:.4f}'.format('IoU@'+str(top_n_list[i])+'@'+str(IoU_threshs[j]), topn_IoU_matric[i,j]/count), end=' | ')
                meters_5['mIoU'].reset()

                print()

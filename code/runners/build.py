import os, sys, glob

def build_model(args, word2vec) : 
    index_to_key = word2vec.index_to_key

    if args.model == 'cmin' :
        from models.cmin import Model 
        model = Model(args)

    elif args.model == 'csmgan' :
        from models.csmgan import Model
        model = Model(args)

    elif args.model == 'tdtan' :
        from models.tdtan import Model
        model = Model()
        
    elif args.model == 'ranet' :
        from models.ranet import Model
        model = Model()

    elif args.model == 'ianet' :
        from models.ianet import Model
        model = Model(args)
    
    elif args.model == 'drn' :
        from models.drn import Model
        model = Model(len(index_to_key), args)
        
    elif args.model == 'tmlga' :
        from models.tmlga import Model
        from modules.tmlga.config import cfg
        model = Model(cfg)
        
    elif args.model == 'reloclnet':
        from models.reloclnet import Model
        model = Model(args)

    elif args.model == 'lvtr':
        from models.lvtr import Model
        model = Model(args)

    elif args.model == 'cnm' : 
        from models.cnm import Model
        model_config = args.cnm
        config = model_config.config
        config['vocab_size'] = args.vocab_size + 1
        config['max_num_words'] = args.max_num_words
        config['frames_input_size'] = args.frame_dim
        config['words_input_size'] = args.word_dim
        config['loss'] = args.loss
       
        model = Model(config)
    
    elif args.model == 'mgpn' :
        from models.mgpn import Model
        model = Model()

    elif args.model == 'plrn' :
        from models.plrn import Model
        model = Model(args)
    
    elif args.model == 'lgi' :
        from models.lgi import Model
        model = Model(args)

    elif args.model == 'tga' :
        from models.tga import Model
        model = Model(args)

    elif args.model == 'trm' :
        from models.trm import Model
        from modules.trm.config import config
        config.merge_from_file(args.cfg)
        args.max_num_epochs = args.SOLVER.MAX_EPOCH
        model = Model(args) 
        
    elif args.model == 'cpl' :
        from models.cpl import Model
        model = Model(args)
    
    elif args.model == 'vslnet' : 
        from models.vslnet import Model
        model = Model(args)
    
    elif args.model == 'msdetr' :
        import torch.nn as nn
        from models.msdetr import Model, QueryBasedDecoder
        from modules.msdetr.detectron2.config.lazy import LazyCall as L
        from modules.msdetr.kn_util.config import eval_str
        from modules.msdetr.backbone import SegFormerX, SegFormerXFPN
        frame_pooler = nn.Identity()
        backbone = SegFormerX(d_model_in=args.d_model,
                               d_model_lvls=[args.d_model] * args.num_layers_enc,
                               num_head_lvls=[args.nhead] * args.num_layers_enc,
                               ff_dim_lvls=[args.d_model] * args.num_layers_enc,
                               input_vid_dim=args.frame_dim,
                               input_txt_dim=args.word_dim,
                               max_vid_len=args.max_len_video,
                               max_txt_len=args.max_num_words,
                               sr_ratio_lvls=args.sr_ratio_lvls,
                               use_patch_merge=args.use_patch_merge,
                               output_layers=[0, 1, 4])
        
        head = QueryBasedDecoder(
            d_model=args.d_model,
            nhead=args.nhead,
            ff_dim=args.ff_dim,
            num_query=args.num_query,
            num_layers=args.num_layers_dec,
            num_scales=3,
            pooler_resolution=4,
        )

        if args.dataset == 'Tacos' and args.with_fpn == True:
            backbone = SegFormerXFPN(backbone=backbone,
                                     output_layer=[0, 2, 3],
                                     intermediate_hidden_size=[args.d_model] * 3,
                                     fpn_hidden_size=args.d_model)
            head = QueryBasedDecoder(
            d_model=args.d_model,
            nhead=args.nhead,
            ff_dim=args.ff_dim,
            num_query=args.num_query,
            num_layers=args.num_layers_dec,
            num_scales=3,
            pooler_resolution=32,
        )

        model = Model(backbone=backbone, head=head, frame_pooler=frame_pooler, model_cfg=args)

    return model

def build_forward(args, batch, model) : 

    if args.model == 'tmlga' :
        index, vid_feats, vid_feats_length, words_vec, words_len, start, end, localiz, localiz_lengths, gt, factors, fps = batch
    else : 
        index, vid_feats, video_mask, words_vec, word_mask, label, \
        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
        sub_feats, sub_mask, tef_feats = batch 
                
    if args.model in ['ianet', 'cmin', 'csmgan'] :    
        model_input = {
            'frames': vid_feats.cuda(),
            'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 'word_mask': word_mask.cuda(),
            'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'fr_label': label.cuda(),
            'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
        } 
        predict_boxes, loss, _ = model(**model_input)

    elif args.model in ['tdtan', 'ranet'] :        
        model_input = {
            'index': index, 'frames': vid_feats.cuda(), 'frame_mask': video_mask.cuda(), 'words': words_vec.cuda(), 
            'word_mask': word_mask.cuda(), 'fr_label': label.cuda(), 'node_pos': id2pos.cuda(), 'map_gt': map_gt.cuda(), 
            'duration': duration.cuda(), 'gt': gt.cuda()
        }
        joint_prob, loss  = model(model_input['words'], model_input['word_mask'], model_input['frames'], model_input['map_gt'])


    elif args.model == 'drn' :
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
        box_lists, loss_dict = model(**model_input)

    elif args.model == 'tmlga' :
        model_input = {
            'videoFeat': vid_feats.cuda(), 'videoFeat_lengths': vid_feats_length.cuda(), 'tokens': words_vec.cuda(),
            'tokens_lengths': words_len.cuda(), 'start': start.cuda(), 'end': end.cuda(), 'localiz': localiz.cuda()
        }
        loss, indvidual_loss, pred_start, pred_end, attention, attention_loss = model(**model_input)


    elif args.model == 'reloclnet' :
        from modules.reloclnet.utils import get_match_labels, get_st_ed_label

        use_video = "video" in args.ctx_mode
        use_sub = "sub" in args.ctx_mode
        use_tef = "tef" in args.ctx_mode

        if use_video and use_tef :
            vid_feats = torch.cat([vid_feats, tef_feats], dim=2)
        if use_sub and use_tef:
            sub_feats = torch.cat([sub_feats, tef_feats], dim=2)
            
        match_labels = torch.tensor(get_match_labels(label, args.max_num_frames))
        model_input = {'query_feat' : words_vec.cuda(), 'query_mask': word_mask.cuda(),
                        'video_feat':vid_feats.cuda(), 'video_mask':video_mask.cuda(),
                        'sub_feat':sub_feats.cuda(), 'sub_mask': sub_mask.cuda(),
                        'st_ed_indices': label.cuda(), 'match_labels': match_labels.cuda()}
        loss, loss_dict = model(**model_input) 

    elif args.model == 'lvtr':
        model_inputs = {'src_txt': words_vec.cuda(non_blocking=args.pin_memory),
                        'src_txt_mask': word_mask.cuda(non_blocking=args.pin_memory),
                        'src_vid': vid_feats.cuda(non_blocking=args.pin_memory),
                        'src_vid_mask': video_mask.cuda(non_blocking=args.pin_memory)}

        targets = {}
        xx_spans = label / (args.num_input_frames-1)
        center = xx_spans.sum(-1) * 0.5
        width = xx_spans[..., 1] - xx_spans[..., 0]
        stack = torch.stack([center, width], dim=-1)
        targets['target_spans'] = [{"spans": v.unsqueeze(0).cuda(non_blocking=args.pin_memory)} for v in stack]

        outputs, loss = model(**model_inputs,
                targets=targets,
                att_visualize=args.att_visualize,
                corr_visualize=args.corr_visualize,
                epoch_i=epoch,
                idx=bid)

    elif args.model == 'tga':
        model_input = {'images': vid_feats.cuda(), 'captions': words_vec.cuda(),
                               'lengths': lengths, 'lengths_img': lengths_img}
        loss = self.model(**model_input)

    elif args.model == 'trm':
        loss = self.model(batch)

    elif args.model == 'msdetr':
        model_input = {'vid_feat' : vid_feats.cuda(), 'txt_feat': words_vec.cuda(), 'gt': gt.cuda(), 'word_mask': word_mask.cuda(), 
                       'word_label': label.cuda(), 'txt_mask': word_mask.cuda(), 'video_id': index, 'text': text}
        losses = model(**model_input)
        loss = losses['loss']

    return model_inputs







import torch

def optimizer_scheduler(optim, sched, param_dicts, lr, wd, lr_drop_step):
    # optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(param_dicts, lr=lr, momentum=0.9, weight_decay=wd)
    if optim == 'adam':
        optimizer = torch.optim.Adam(param_dicts, lr=lr, weight_decay=wd)
    if optim == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=wd)

    # scheduler
    if sched == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_step) #, gamma=gamma
    if sched == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_size=lr_drop_step) #, gamma=gamma
    if sched == 'reducelronplateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, threshold=0.5, verbose=True)

    return optimizer, scheduler
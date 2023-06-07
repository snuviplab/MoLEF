from scipy.stats import gaussian_kde
import numpy as np
from utils import load_json, load_pickle

def estimate_maxima(data):
    kde = gaussian_kde(data)
    no_samples = 100
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]

    return maxima

def ensemble(*output) :
    '''
    *output: variable dictionary 
            keys - idx, predict_starts, predict_ends, gt_starts, gt_ends 
    '''
    output_starts = np.array([])
    output_ends = np.array([])
    for i in range(len(output)) :
        predict = load_pickle(output[i])
        if i == 0 : 
            output_starts = np.hstack((output_starts, predict['predict_starts']))
            output_ends = np.hstack((output_ends, predict['predict_ends']))
        else :     
            output_starts = np.vstack((output_starts, predict['predict_starts']))
            output_ends = np.vstack((output_ends, predict['predict_ends']))

    pred_starts = []
    pred_ends = []
    for i in range(len(output_starts[0])) :
        try : 
            pred_start_max = estimate_maxima(np.array(output_starts[:,i]))
            pred_end_max = estimate_maxima(np.array(output_ends[:,i]))
        except:
            scounter = Counter(output_starts[:, i])
            ecounter = Counter(output_ends[:, i])
            pred_start_max = scounter.most_common(1)[0][0]
            pred_end_max = ecounter.most_common(1)[0][0]
        pred_starts.append(pred_start_max)
        pred_ends.append(pred_end_max)
     
    return output[0][idx], (output_starts, output_ends), (pred_starts, pred_ends), (output[0]['gt_starts'], output[0]['gt_ends'])


if __name__ == "__main__" :

    # initialize 
    paths = "/data/projects/VT_localization/jinyeong/TSGV/code/utils/ensemble/paths.yaml"
    pickles = []
    model_names = []
    for k, v in paths['Activitynet'].items() :
        if v : 
            pickles.append(v)
            model_names.append(k)
    model_names.append("ensemble")
    table_cols = ["model", "mIoU", "R@1, IoU=0.1 ", "R@1, IoU=0.3 ", "R@1, IoU=0.5 ", "R@1, IoU=0.7 ", "R@1, IoU=0.9 "]
   
    # calculate output and ensemble 
    idx, output, ensemble, gt = ensemble(pickles)
    
    # calculate IoUs
    total_ious = []
    for i in range(len(model_names)) : 
        IoUs = calculate_IoU_batch((output[0][i], output[1][i]), (gt[0], gt[1]))
        total_ious.append(IoUs)
    IoUs = calculate_IoU_batch((ensemble[0], ensemble[1]), (gt[0], gt[1]))
    total_ious.append(IoUs)

    # transform recall and round IoUs 
    table_data = []
    for name, iou in zip(model_names, total_ious) :
        temp = []
        temp.append(name)
        temp.append(np.round(np.mean(iou)*100, 2))
        for i in range(1, 10, 2) :
            temp.append(np.round(np.mean(iou >= (i / 10))*100, 2))
        table_data.append(temp)
    
    # final 
    table = pd.DataFrame(table_data, columns=table_cols)





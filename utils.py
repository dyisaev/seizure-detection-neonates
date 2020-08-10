import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import keras

def rearrange_epochs(all_feats, annot, epoch_type, annot_chan=None, n_chan=12):
    '''
    epoch_type=0 for n_chan x 55 chunks of features by Temko et al.
    epoch_type=1 for n_chan x 256 chunks of downsampled data
    '''

    ann_pool, ann_chan = rearrange_annot(annot, annot_chan)
    epochs_lst = []
    for i in range(n_chan):
        cur_epoch = all_feats[0, 0][i, 0][0, 0][epoch_type]
        cur_epoch = cur_epoch.ravel()
        elem_lst = []
        for elem in cur_epoch:
            elem = elem.ravel()
            elem_lst.append(elem)
        upd_epoch = np.stack(elem_lst, axis=0)
        epochs_lst.append(upd_epoch)
    epoch_np = np.stack(epochs_lst, axis=0)
    epoch_np = epoch_np.swapaxes(0, 1)
    if epoch_np.shape[0] != ann_pool.shape[0]:
        print('epochs do not correspond to annot!')
        return []
    idx_arr = np.ones(epoch_np.shape[0]) == 1
    for idx in range(epoch_np.shape[0]):
        if not np.all(np.isfinite(epoch_np[idx, :, :])):
            idx_arr[idx] = 0
            continue
        # standardize epoch data

    print('NaN epochs:', np.sum(idx_arr == 0))
    print('correct epochs:', np.sum(idx_arr == 1))
    if not annot_chan is None:
        return epoch_np[idx_arr], ann_pool[idx_arr], ann_chan[idx_arr],idx_arr
    else:
        return epoch_np[idx_arr], ann_pool[idx_arr], idx_arr


def rearrange_annot(annot, annot_chan=None, poolsize=8, step=4, thr=0.8):
    ann_chan_lst = []
    annot = annot.ravel()
    ann_pool_lst = []
    i = 0
    k = 0
    while (i + poolsize) < annot.shape[0]:
        ann_pool_lst.append(np.mean(annot[i:i + poolsize]))
        if annot_chan is not None:
            ann_chan_lst.append(
                np.mean(annot_chan[i:i + poolsize], axis=0))  # Channels that were active for more than a half of chunk
        i += step
        k += 1
    ann_pool = np.asarray(ann_pool_lst)
    ann_pool = ann_pool > thr
    if annot_chan is not None:
        ann_chan = np.asarray(ann_chan_lst)
        ann_chan = ann_chan > thr
    else:
        ann_chan = None
    return ann_pool, ann_chan

"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""
import sys

import os
import numpy as np
import pandas as pd
import json
import shutil
import keras
from collections import defaultdict
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)

from tornet.data.loader import get_dataloader
from tornet.data.preprocess import get_shape
from tornet.data.constants import ALL_VARIABLES

from tornet.models.keras.losses import mae_loss

from tornet.models.keras.cnn_baseline import build_model

from tornet.metrics.keras import metrics as tfm

from tornet.utils.general import make_exp_dir, make_callback_dirs

EXP_DIR=os.environ.get('EXP_DIR','.')
DATA_ROOT=os.environ['TORNET_ROOT']
logging.info('TORNET_ROOT='+DATA_ROOT)

DEFAULT_CONFIG={
    'epochs':10,
    'input_variables':ALL_VARIABLES,
    # 'train_years':list(range(2013,2021)),
    # 'val_years':list(range(2021,2023)),
    'train_years':list(range(2013)),
    'val_years':list(range(2013)),
    'batch_size':128,
    'model':'vgg',
    'start_filters':48,
    'learning_rate':1e-4,
    'decay_steps':1386,
    'decay_rate':0.958,
    'l2_reg':1e-5,
    'wN':1.0,
    'w0':1.0,
    'w1':1.0,
    'w2':2.0,
    'wW':0.5,
    'label_smooth':0,
    'loss':'cce',
    'head':'maxpool',
    'exp_name':'tornet_baseline',
    'exp_dir':EXP_DIR,
    'dataloader':"keras",
    'dataloader_kwargs': {},
    'use_madis_data': False
}

def main(config):
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    learning_rate=config.get('learning_rate')
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
    l2_reg=config.get('l2_reg')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    head=config.get('head')
    label_smooth=config.get('label_smooth')
    loss_fn = config.get('loss')
    input_variables=config.get('input_variables')
    exp_name=config.get('exp_name')
    exp_dir=config.get('exp_dir')
    train_years=config.get('train_years')
    val_years=config.get('val_years')
    dataloader=config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')
    use_madis_data = config.get('use_madis_data')
    catalog_path = config.get('catalog_path', None)
    max_files = config.get('max_files', None)

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f'Using {dataloader} dataloader')
    logging.info('Running with config:')
    logging.info(config)

    weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}

    # Load pre-filtered catalog if provided (ensures both MADIS and no-MADIS runs see identical storms)
    catalog_df = None
    if catalog_path:
        catalog_df = pd.read_csv(catalog_path, parse_dates=['start_time', 'end_time'])
        logging.info(f'Using pre-filtered catalog: {catalog_path} ({len(catalog_df)} rows)')

    # Create data loaders
    select_keys = input_variables + ['range_folded_mask', 'coordinates']
    if use_madis_data:
        select_keys.append('madis')
        dataloader_kwargs.update({'use_madis_data': True})
    dataloader_kwargs.update({'select_keys': select_keys})
    # Pass catalog_df and max_files directly (not via dataloader_kwargs to avoid JSON serialization issues)
    extra_kwargs = {}
    if catalog_df is not None:
        extra_kwargs['catalog'] = catalog_df
    if max_files is not None:
        extra_kwargs['max_files'] = max_files
    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs, **extra_kwargs)
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, weights, **dataloader_kwargs, **extra_kwargs)
    
    # Load first batch directly (avoids TF iterator which can deadlock)
    first_batch = ds_train[0]
    x = first_batch[0]
    # Use concrete shapes for MLP head (needed for proper shape inference with concatenation)
    if head == 'mlp':
        in_shapes = tuple(x[input_variables[0]].shape[1:])  # (height, width, tilts)
        c_shapes = tuple(x["coordinates"].shape[1:])
    else:
        in_shapes = (None, None, get_shape(x)[-1])
        c_shapes = (None, None, x["coordinates"].shape[-1])
    
    nn = build_model(shape=in_shapes,
                     c_shape=c_shapes,
                     start_filters=start_filters,
                     l2_reg=l2_reg,
                     input_variables=input_variables,
                     head=head,
                     use_madis=use_madis_data)
    
    # model setup
    lr=keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase=False, name="exp_decay")
    
    from_logits=True
    if loss_fn.lower()=='cce':
        loss = keras.losses.BinaryCrossentropy( from_logits=from_logits, 
                                                    label_smoothing=label_smooth )
    elif loss_fn.lower()=='hinge':
        loss = keras.losses.Hinge() # automatically converts labels to -1,1
    elif loss_fn.lower()=='mae':
        loss = lambda yt,yp: mae_loss(yt,yp)
    else:
        raise RuntimeError('unknown loss %s' % loss_fn)


    opt  = keras.optimizers.Adam(learning_rate=lr)

    # Compute various metrics while training
    metrics = [keras.metrics.AUC(from_logits=from_logits,name='AUC',num_thresholds=2000),
                keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000), 
                tfm.BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                tfm.TruePositives(from_logits,name='TruePositives'),
                tfm.FalsePositives(from_logits,name='FalsePositives'), 
                tfm.TrueNegatives(from_logits,name='TrueNegatives'),
                tfm.FalseNegatives(from_logits,name='FalseNegatives'), 
                tfm.Precision(from_logits,name='Precision'), 
                tfm.Recall(from_logits,name='Recall'),
                tfm.F1Score(from_logits=from_logits,name='F1')]
    
    nn.compile(loss=loss,
                metrics=metrics,
                optimizer=opt,
                weighted_metrics=[],
                run_eagerly=True)
    
    ## Setup experiment directory and model callbacks
    expdir = make_exp_dir(exp_dir=exp_dir,prefix=exp_name)
    logging.info('expdir='+expdir)

    # Copy the properties that were used
    with open(os.path.join(expdir,'data.json'),'w') as f:
        json.dump(
            {'data_root':DATA_ROOT,
             'train_data':list(train_years), 
             'val_data':list(val_years)},f)
    with open(os.path.join(expdir,'params.json'),'w') as f:
        json.dump({'config':config},f)
    # Copy the training script
    shutil.copy(__file__, os.path.join(expdir,'train.py')) 
    
    # Create callback directories (checkpoints_dir used by manual loop)
    _tboard_dir, checkpoints_dir = make_callback_dirs(expdir)

    ## FIT — manual loop bypasses model.fit() to avoid TF threading deadlock with xarray/HDF5
    history_dict = _manual_train_loop(nn, ds_train, ds_val, epochs, expdir, checkpoints_dir)

    # At the end, report the best score observed over all epochs
    val_aucs = history_dict.get('val_AUC', [])
    val_aucprs = history_dict.get('val_AUCPR', [])
    best_auc = float(np.max(val_aucs)) if val_aucs else 0.5
    best_aucpr = float(np.max(val_aucprs)) if val_aucprs else 0.0

    return {'AUC': best_auc, 'AUCPR': best_aucpr}


def _manual_train_loop(model, ds_train, ds_val, epochs, expdir, checkpoints_dir):
    """Manual epoch/batch loop that replaces model.fit().

    Iterates ds_train and ds_val via __getitem__ so Python—not TF's C++
    data pipeline—drives I/O, which prevents the threading deadlock that
    occurs when xarray opens HDF5 files inside TF's background threads.
    """
    csv_path = os.path.join(expdir, 'history.csv')
    history = defaultdict(list)

    rng = np.random.default_rng(seed=1234)
    for epoch in range(epochs):
        logging.info(f'Epoch {epoch + 1}/{epochs}')

        # Shuffle file list each epoch (mirrors model.fit() default behavior)
        rng.shuffle(ds_train.file_list)

        # ---- Training ----
        model.reset_metrics()
        result = {}
        with tqdm(range(len(ds_train)), desc=f'Epoch {epoch+1}/{epochs} train', unit='batch', leave=False) as pbar:
            for i in pbar:
                batch = ds_train[i]
                if len(batch) == 3:
                    x, y, w = batch
                    result = model.train_on_batch(x, y, sample_weight=w, return_dict=True)
                else:
                    x, y = batch
                    result = model.train_on_batch(x, y, return_dict=True)
                pbar.set_postfix(loss=f"{result.get('loss', float('nan')):.4f}",
                                 AUC=f"{result.get('AUC', float('nan')):.4f}")

        train_results = {k: float(v) for k, v in result.items()}

        # ---- Validation ----
        model.reset_metrics()
        vres = {}
        with tqdm(range(len(ds_val)), desc=f'Epoch {epoch+1}/{epochs} val  ', unit='batch', leave=False) as pbar:
            for i in pbar:
                batch = ds_val[i]
                if len(batch) == 3:
                    x, y, w = batch
                    vres = model.test_on_batch(x, y, sample_weight=w, return_dict=True)
                else:
                    x, y = batch
                    vres = model.test_on_batch(x, y, return_dict=True)
                pbar.set_postfix(loss=f"{vres.get('loss', float('nan')):.4f}")

        val_results = {k: float(v) for k, v in vres.items()}

        # ---- Log epoch summary ----
        summary = '  '.join([f'{k}: {v:.4f}' for k, v in train_results.items()])
        summary += '  ' + '  '.join([f'val_{k}: {v:.4f}' for k, v in val_results.items()])
        logging.info(f'Epoch {epoch + 1}/{epochs} — {summary}')

        # ---- Accumulate history ----
        for k, v in train_results.items():
            history[k].append(v)
        for k, v in val_results.items():
            history[f'val_{k}'].append(v)

        # ---- Save history CSV ----
        pd.DataFrame(dict(history)).to_csv(csv_path, index=False)

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(checkpoints_dir, f'tornadoDetector_{epoch + 1:03d}.keras')
        model.save(ckpt_path)
        logging.info(f'Saved checkpoint: {ckpt_path}')

    return dict(history)


if __name__=='__main__':
    config=DEFAULT_CONFIG
    # Load param file if given
    if len(sys.argv)>1:
        config.update(json.load(open(sys.argv[1],'r')))
    main(config)

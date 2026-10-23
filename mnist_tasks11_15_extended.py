"""NumPy MNIST experiments based on 'Ablation Studies & Experiments for NumPy MNIST Neural Network'.

Usage (put this script beside mnist_train.csv and mnist_test.csv):
    python mnist_tasks11_15_extended.py --list
    python mnist_tasks11_15_extended.py --run baseline
    python mnist_tasks11_15_extended.py --run activations
    python mnist_tasks11_15_extended.py --run residual
    python mnist_tasks11_15_extended.py --run error_analysis
    python mnist_tasks11_15_extended.py --run all --quick

A full set of experiments can require many hours. The default runs only the baseline.
This is a teaching implementation, using NumPy rather than PyTorch/TensorFlow.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save plots; works in Windows terminals without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / 'Results'

# Your original Stage 2 dictionary. All options below work in this file.
config = {
    'layer_sizes': [784, 128, 10],
    'activations': ['relu', 'softmax'],
    'init': 'he',
    'lr': 0.01,
    'batch_size': 64,
    'epochs': 20,
    'optimizer': 'sgd',
    'loss': 'cross_entropy',
    'weight_decay': 0.0,
    'dropout': 0.0,
    'seed': 42,
    # Extensions requested in later PDF sections
    'preprocessing': 'zero_one',  # raw, zero_one, minus_one_one, standardize
    'architecture': 'plain',    # plain, residual, multi_branch
    'branch_widths': [128, 128],
    'batch_norm': False,
    'augment': False,
    'early_stopping': False,
    'patience': 3,
    'schedule': 'constant',     # constant, step, exponential, cosine
}

HIDDEN = ('relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'swish', 'linear')
OPTS = ('sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam')
MODES = (
    'baseline', 'activations', 'sigmoid_depth', 'depth', 'width', 'learning_rate',
    'lr_schedules', 'batch_size', 'batch_scaling', 'optimizers',
    'initialization', 'regularization', 'loss_functions', 'preprocessing',
    'residual', 'bottleneck', 'multi_branch', 'output_activation',
    'tuning', 'debugging', 'error_analysis', 'batch_norm',
    'mc_dropout', 'ensemble', 'adversarial', 'all',
)
QUESTIONS = {
    'activations': ['Which activation gives the best accuracy and convergence?',
                    'How do activation derivatives explain vanishing gradients?',
                    'What happens with sigmoid in a five-layer network?'],
    'sigmoid_depth': ['How does five hidden layers with sigmoid compare with ReLU?',
                      'Do the activation derivatives help explain the results?'],
    'depth': ['Does deeper always mean better?', 'How do gradient norms change by depth?',
              'How does depth affect overfitting?'],
    'width': ['What is the best width for MNIST?', 'How does width affect loss smoothness?',
              'Compare parameter count, time and accuracy.'],
    'learning_rate': ['Which rates diverge or oscillate?', 'How does batch size interact with learning rate?',
                      'Which rates reach 90% validation accuracy fastest?'],
    'lr_schedules': ['Which learning-rate schedule gives the best final validation accuracy?'],
    'batch_size': ['Why can small batches regularize?', 'How many weight updates occur per epoch?',
                   'How do batch sizes affect runtime and memory?'],
    'batch_scaling': ['Does doubling batch size and learning rate preserve convergence speed?'],
    'optimizers': ['Which optimizers are robust to learning rate?', 'How do adaptive optimizers rescale updates?',
                   'Why might Adam generalize differently from SGD?'],
    'initialization': ['Why does zero initialization prevent hidden neurons from specializing?',
                       'How do initializations affect gradient and activation distributions?'],
    'regularization': ['Which approach reduces the training-validation gap?',
                       'How do you choose regularization strength?', 'Why is dropout similar to ensembling?'],
    'loss_functions': ['Why is cross-entropy typically preferred for classification?',
                       'Compare softmax-cross-entropy and softmax-MSE gradients.'],
    'preprocessing': ['Why does normalization affect optimization?',
                      'How do pixel means and standard deviations change with standardization?'],
    'residual': ['Do skip connections improve deep MLP training on MNIST?'],
    'bottleneck': ['Does the narrow middle layer help or hurt performance?'],
    'multi_branch': ['Does a two-branch architecture improve test accuracy and training speed?'],
    'output_activation': ['How does softmax differ from one-vs-all sigmoid?'],
    'tuning': ['Which hyperparameters matter most?', 'How many random trials match grid search?',
               'Why does a large search space make tuning difficult?'],
    'debugging': ['Do analytical and numerical gradients agree?',
                  'Can the network reach 100% training accuracy on 10 images?',
                  'What do the first-layer weights represent?'],
    'error_analysis': ['Which digits are most confused?', 'Are errors handwriting ambiguity or model limits?',
                       'How would class imbalance affect the metrics?'],
    'batch_norm': ['Does batch normalization improve training speed or stability?'],
    'mc_dropout': ['How uncertain are repeated dropout predictions?'],
    'ensemble': ['Does combining five independently seeded models improve accuracy?'],
    'adversarial': ['How does accuracy change after an FGSM perturbation?'],
}


def validate(c):
    s, a = c['layer_sizes'], c['activations']
    if len(s) < 2 or any(int(x) != x or x < 1 for x in s):
        raise ValueError('layer_sizes must contain positive integers')
    if len(a) != len(s) - 1 or any(x not in HIDDEN for x in a[:-1]):
        raise ValueError('One supported activation is required per non-input layer')
    if a[-1] not in ('softmax', 'sigmoid'):
        raise ValueError('The output activation must be softmax or sigmoid')
    if c['optimizer'] not in OPTS or c['loss'] not in ('cross_entropy', 'mse'):
        raise ValueError('Unknown optimizer/loss')
    if c['init'] not in ('zeros', 'small', 'large', 'xavier', 'he'):
        raise ValueError('Unknown initialization')
    if c['preprocessing'] not in ('raw', 'zero_one', 'minus_one_one', 'standardize'):
        raise ValueError('Unknown preprocessing')
    if c['architecture'] not in ('plain', 'residual', 'multi_branch'):
        raise ValueError('Unknown architecture')
    if c['schedule'] not in ('constant', 'step', 'exponential', 'cosine'):
        raise ValueError('Unknown learning-rate schedule')
    if not (0 <= c['dropout'] < 1) or c['weight_decay'] < 0:
        raise ValueError('Invalid dropout / weight_decay')
    if c['lr'] <= 0 or c['epochs'] < 1 or c['batch_size'] < 1:
        raise ValueError('Learning rate, epochs, and batch size must be positive')
    if c['architecture'] == 'multi_branch' and (len(c['branch_widths']) != 2 or
                                                min(c['branch_widths']) <= 0):
        raise ValueError('Two positive branch widths required')
    if c['architecture'] == 'multi_branch' and len(s) != 3:
        raise ValueError('For multi_branch use layer_sizes=[784, 128, 10]')
    if c['augment'] and s[0] != 784:
        raise ValueError('Image augmentation requires 28x28 input')
    if c['early_stopping'] and c['patience'] < 1:
        raise ValueError('patience must be >=1')


def read_csv(filename):
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Place the CSV beside this Python file.')
    # Header=None correctly retains the first image for headerless MNIST CSVs.
    data = pd.read_csv(path, header=None)
    if pd.to_numeric(data.iloc[:1, 0], errors='coerce').isna().any():
        data = data.iloc[1:]
    arr = data.to_numpy(dtype=np.float32)
    if arr.shape[1] != 785:
        raise ValueError(f'{filename}: expected 785 columns (label + 784 pixels), got {arr.shape[1]}')
    y = arr[:, 0].astype(np.int64)
    if np.any((y < 0) | (y > 9)):
        raise ValueError(f'{filename}: digit labels must be 0 through 9')
    return arr[:, 1:].T.copy(), y


def load_data(seed=42):
    print('Loading MNIST CSVs from', ROOT, flush=True)
    X, y = read_csv('mnist_train.csv')
    T, yt = read_csv('mnist_test.csv')
    if len(y) < 12:
        raise ValueError('At least 12 training examples needed for a validation split')
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(y))
    nv = min(10000, max(1, len(y) // 6))
    # Hold-out validation. The test set is never used for updating weights.
    return {
        'train': (X[:, ids[nv:]], y[ids[nv:]]),
        'val': (X[:, ids[:nv]], y[ids[:nv]]),
        'test': (T, yt),
    }


def subset_data(data, ntrain=None, nval=None, ntest=None):
    out = {}
    for part, limit in [('train', ntrain), ('val', nval), ('test', ntest)]:
        X, y = data[part]
        n = len(y) if limit is None else min(len(y), limit)
        out[part] = X[:, :n], y[:n]
    return out


def preprocess(data, mode):
    train_x = data['train'][0]
    mu = train_x.mean(axis=1, keepdims=True) if mode == 'standardize' else None
    sd = train_x.std(axis=1, keepdims=True) if mode == 'standardize' else None
    if sd is not None:
        sd = np.maximum(sd, 1e-5)  # Completely blank pixels have zero std.
    out = {}
    for split, (X, y) in data.items():
        if mode == 'raw':
            x = X.copy()
        elif mode == 'zero_one':
            x = X / 255.0
        elif mode == 'minus_one_one':
            x = X / 127.5 - 1.0
        else:
            x = (X - mu) / sd
        out[split] = (np.asarray(x, dtype=np.float32), y)
    return out


def sigmoid(z):
    # Numerically stable for large positive/negative values.
    z = np.clip(z, -80, 80)
    return 1.0 / (1.0 + np.exp(-z))


def act(z, name):
    if name == 'relu': return np.maximum(0, z)
    if name == 'sigmoid': return sigmoid(z)
    if name == 'tanh': return np.tanh(z)
    if name == 'leaky_relu': return np.where(z > 0, z, 0.01*z)
    if name == 'elu': return np.where(z > 0, z, np.expm1(np.minimum(z, 0)))
    if name == 'swish': return z * sigmoid(z)
    if name == 'linear': return z
    if name == 'softmax':
        ex = np.exp(z - np.max(z, axis=0, keepdims=True))
        return ex / np.sum(ex, axis=0, keepdims=True)
    raise ValueError(name)


def d_act(z, name):
    if name == 'relu': return (z > 0).astype(z.dtype)
    if name == 'sigmoid':
        s = sigmoid(z)
        return s * (1-s)
    if name == 'tanh': return 1-np.tanh(z)**2
    if name == 'leaky_relu': return np.where(z > 0, 1.0, 0.01)
    if name == 'elu': return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))
    if name == 'swish':
        s = sigmoid(z)
        return s + z*s*(1-s)
    if name == 'linear': return np.ones_like(z)
    raise ValueError(name)


def target(y, k):
    v = np.zeros((k, len(y)), dtype=np.float32)
    v[y, np.arange(len(y))] = 1
    return v


def calc_loss(A, y, loss_name, output_act, with_gradient=False):
    """Return loss and dL/dZ_output (including 1/batch_size)."""
    m = len(y)
    Y = target(y, A.shape[0]).astype(A.dtype)
    if loss_name == 'cross_entropy':
        if output_act == 'softmax':
            v = np.clip(A[y, np.arange(m)], 1e-12, 1)
            loss = -np.log(v).mean()
        else:  # Sum of 10 binary cross-entropies (one-vs-all).
            B = np.clip(A, 1e-12, 1-1e-12)
            loss = -np.sum(Y*np.log(B) + (1-Y)*np.log(1-B))/m
        dz = (A - Y) / m
    else:  # 0.5 * mean(sum over 10 classes of squared error)
        d = A-Y
        loss = 0.5*np.sum(d*d)/m
        dA = d/m
        if output_act == 'softmax':
            dz = A * (dA - np.sum(A*dA, axis=0, keepdims=True))
        else:
            dz = dA*A*(1-A)
    return (float(loss), dz) if with_gradient else float(loss)


class Network:
    def __init__(self, cfg, dtype=np.float32):
        validate(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.dtype = dtype
        self.rng = np.random.default_rng(cfg['seed'])
        self.layers = []
        self.bn_running = []
        self.arch = cfg['architecture']
        sizes = cfg['layer_sizes']
        if self.arch == 'multi_branch':
            w1, w2 = cfg['branch_widths']
            dimensions = [(sizes[0], w1), (sizes[0], w2), (w1+w2, sizes[-1])]
        else:
            dimensions = list(zip(sizes[:-1], sizes[1:]))
        for i, (nin, nout) in enumerate(dimensions):
            method = cfg['init']
            scale = {'he': np.sqrt(2/nin), 'xavier': np.sqrt(2/(nin+nout)),
                     'small': 0.01, 'large': 1.0, 'zeros': 0.0}[method]
            p = {'W': (self.rng.standard_normal((nout, nin))*scale).astype(dtype),
                 'b': np.zeros((nout, 1), dtype=dtype)}
            if cfg['batch_norm'] and i < len(dimensions)-1:
                p['gamma'] = np.ones((nout, 1), dtype=dtype)
                p['beta'] = np.zeros((nout, 1), dtype=dtype)
            self.layers.append(p)
            self.bn_running.append({'mean': np.zeros((nout, 1), dtype=dtype),
                                    'var': np.ones((nout, 1), dtype=dtype)})

    def _layer(self, i, x, activation, training, dropout_enabled):
        p = self.layers[i]
        z = p['W']@x + p['b']
        bn = None
        h = z
        if 'gamma' in p:
            if training:
                mu = z.mean(axis=1, keepdims=True)
                var = z.var(axis=1, keepdims=True)
                r = self.bn_running[i]
                r['mean'][:] = 0.9*r['mean'] + 0.1*mu
                r['var'][:] = 0.9*r['var'] + 0.1*var
            else:
                r = self.bn_running[i]
                mu, var = r['mean'], r['var']
            inv = 1/np.sqrt(var + 1e-5)
            xhat = (z-mu)*inv
            h = p['gamma']*xhat + p['beta']
            bn = {'xhat': xhat, 'inv': inv, 'training': training}
        a = act(h, activation)
        mask = None
        if dropout_enabled and self.cfg['dropout']:
            d = self.cfg['dropout']
            mask = (self.rng.random(a.shape) >= d).astype(a.dtype)/(1-d)
            a = a*mask
        return a, {'input': x, 'h': h, 'bn': bn, 'mask': mask, 'activation': activation,
                   'skip': None}

    def forward(self, X, training=False, mc_dropout=False, return_cache=False):
        acts = self.cfg['activations']
        drop = training or mc_dropout
        if self.arch == 'multi_branch':
            a0, c0 = self._layer(0, X, acts[0], training, drop)
            a1, c1 = self._layer(1, X, acts[0], training, drop)
            a2, c2 = self._layer(2, np.vstack((a0, a1)), acts[-1], training, False)
            return (a2, {'acts': [X, a0, a1, a2], 'caches': [c0,c1,c2]}) if return_cache else a2
        a = X
        values = [X]
        caches = []
        for i, name in enumerate(acts):
            hidden = i < len(acts)-1
            a, cache = self._layer(i, a, name, training, drop and hidden)
            # Every two hidden layers, add an identity shortcut where shapes match.
            if self.arch == 'residual' and hidden and i >= 2 and i % 2 == 0:
                previous = values[i-1]  # Output of hidden layer i-2.
                if previous.shape == a.shape:
                    a = a + previous
                    cache['skip'] = i-2
            values.append(a)
            caches.append(cache)
        return (a, {'acts': values, 'caches': caches}) if return_cache else a

    def _bn_back(self, i, dh, bn):
        p = self.layers[i]
        g = {'gamma': np.sum(dh*bn['xhat'], axis=1, keepdims=True),
             'beta': np.sum(dh, axis=1, keepdims=True)}
        dxhat = dh*p['gamma']
        if bn['training']:
            m = dh.shape[1]
            dz = bn['inv']*(dxhat-dxhat.sum(axis=1, keepdims=True)/m -
                             bn['xhat']*(dxhat*bn['xhat']).sum(axis=1, keepdims=True)/m)
        else:
            dz = dxhat*bn['inv']
        return dz, g

    def backward(self, A, cache, y, input_gradient=False):
        c = self.cfg
        loss, dout = calc_loss(A, y, c['loss'], c['activations'][-1], True)
        grads = [None]*len(self.layers)
        if self.arch == 'multi_branch':
            layers = cache['caches']
            x, a0, a1, _ = cache['acts']
            dW = dout @ layers[2]['input'].T
            db = dout.sum(axis=1, keepdims=True)
            grads[2] = {'W': dW, 'b': db}
            dcat = self.layers[2]['W'].T@dout
            offset = 0
            dX = np.zeros_like(x)
            for i, a in enumerate([a0, a1]):
                width = a.shape[0]
                d = dcat[offset:offset+width]
                offset += width
                if layers[i]['mask'] is not None: d = d*layers[i]['mask']
                dh = d*d_act(layers[i]['h'], layers[i]['activation'])
                g = {}
                if layers[i]['bn'] is not None:
                    dh, g = self._bn_back(i, dh, layers[i]['bn'])
                grads[i] = {'W': dh@x.T, 'b': dh.sum(axis=1, keepdims=True), **g}
                dX += self.layers[i]['W'].T@dh
        else:
            values = cache['acts']
            entries = cache['caches']
            dA = [np.zeros_like(v) for v in values]
            for i in reversed(range(len(self.layers))):
                ent = entries[i]
                if i == len(self.layers)-1:
                    dz = dout
                    extra = {}
                else:
                    da = dA[i+1]
                    if ent['skip'] is not None:
                        dA[ent['skip']+1] += da  # Gradient travels along the shortcut.
                    if ent['mask'] is not None: da = da*ent['mask']
                    dz = da*d_act(ent['h'], ent['activation'])
                    extra = {}
                    if ent['bn'] is not None:
                        dz, extra = self._bn_back(i, dz, ent['bn'])
                grads[i] = {'W': dz@values[i].T,
                            'b': dz.sum(axis=1, keepdims=True), **extra}
                dA[i] += self.layers[i]['W'].T@dz
            dX = dA[0]
        for i, p in enumerate(self.layers):
            grads[i]['W'] += c['weight_decay']*p['W']
            loss += 0.5*c['weight_decay']*float(np.sum(p['W']*p['W']))
        return (loss, grads, dX) if input_gradient else (loss, grads)

    def count_parameters(self):
        return sum(v.size for p in self.layers for v in p.values())


class Optimizer:
    def __init__(self, net):
        self.name = net.cfg['optimizer']
        self.params = [(i, k, v) for i, p in enumerate(net.layers) for k, v in p.items()]
        self.state = {}
        self.t = 0
        for i, k, v in self.params:
            self.state[(i,k)] = {'velocity': np.zeros_like(v),
                                 'v': np.zeros_like(v), 'm': np.zeros_like(v)}

    def step(self, grads, lr):
        self.t += 1
        eps = 1e-8
        for i, k, p in self.params:
            g = grads[i][k]
            s = self.state[(i,k)]
            if self.name == 'sgd':
                upd = g
            elif self.name in ('momentum', 'nesterov'):
                s['velocity'][:] = 0.9*s['velocity'] + g
                # Nesterov-style lookahead correction, with SGD-compatible velocity definition.
                upd = 0.9*s['velocity'] + g if self.name == 'nesterov' else s['velocity']
            elif self.name == 'adagrad':
                s['v'] += g*g
                upd = g/(np.sqrt(s['v'])+eps)
            elif self.name == 'rmsprop':
                s['v'][:] = 0.9*s['v'] + 0.1*g*g
                upd = g/(np.sqrt(s['v'])+eps)
            elif self.name == 'adam':
                s['m'][:] = 0.9*s['m']+0.1*g
                s['v'][:] = 0.999*s['v']+0.001*g*g
                mh = s['m']/(1-0.9**self.t)
                vh = s['v']/(1-0.999**self.t)
                upd = mh/(np.sqrt(vh)+eps)
            p -= lr*upd


def augment_batch(X, rng, fill=0.0):
    """Random +-2-pixel shifts and +-10-degree rotations with bilinear interpolation.
    X has shape (784, batch), values already preprocessed.
    """
    b = X.shape[1]
    images = X.T.reshape(b, 28, 28)
    out = np.full((b, 28, 28), fill, dtype=X.dtype)
    yy, xx = np.mgrid[0:28, 0:28]
    cx = cy = 13.5
    for k in range(b):
        ang = rng.uniform(-10, 10)*np.pi/180
        dx, dy = rng.integers(-2, 3, size=2)
        u, v = xx-cx-dx, yy-cy-dy
        xs = np.cos(ang)*u + np.sin(ang)*v+cx
        ys = -np.sin(ang)*u + np.cos(ang)*v+cy
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs-x0, ys-y0
        ans = np.zeros((28, 28), dtype=X.dtype)
        for ox, oy, w in [(0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)),
                          (0,1,(1-fx)*fy), (1,1,fx*fy)]:
            xi, yi = x0+ox, y0+oy
            mask = (xi>=0)&(xi<28)&(yi>=0)&(yi<28)
            vals = np.full((28,28), fill, dtype=X.dtype)
            vals[mask] = images[k, yi[mask], xi[mask]]
            ans += vals*w
        out[k] = ans
    return out.reshape(b, 784).T.copy()


def lr_at_epoch(c, epoch):
    lr = c['lr']
    mode = c['schedule']
    if mode == 'step': return lr*0.5**(epoch//5)
    if mode == 'exponential': return lr*0.9**epoch
    if mode == 'cosine':
        return 1e-5 + 0.5*(lr-1e-5)*(1+np.cos(np.pi*epoch/max(1,c['epochs']-1)))
    return lr


def evaluate(net, X, y, cfg, batch=1024, probabilities=False):
    n = len(y)
    total_loss, correct = 0.0, 0
    preds, probs = [], []
    for start in range(0, n, batch):
        x = X[:, start:start+batch]
        Y = y[start:start+batch]
        a = net.forward(x, training=False)
        loss = calc_loss(a, Y, cfg['loss'], cfg['activations'][-1])
        total_loss += loss*len(Y)
        p = np.argmax(a, axis=0)
        correct += np.sum(p == Y)
        if probabilities:
            preds.append(p)
            probs.append(a.T.copy())
    result = {'loss': total_loss/max(1, n), 'accuracy': correct/max(1,n)}
    if probabilities:
        result['predictions'] = np.concatenate(preds)
        result['probabilities'] = np.concatenate(probs)
    return result


def training_time(net, data, run_dir, quiet=False):
    c = net.cfg
    X, y = data['train']
    xv, yv = data['val']
    opt = Optimizer(net)
    rng = np.random.default_rng(c['seed']+1234)
    rows = []
    best_val = float('inf')
    best_state = None
    bad_epochs = 0
    start_total = time.perf_counter()
    for epoch in range(c['epochs']):
        start_epoch = time.perf_counter()
        order = rng.permutation(len(y))
        gmeans = np.zeros(len(net.layers))
        gnorms = np.zeros(len(net.layers))
        batches = 0
        for st in range(0, len(y), c['batch_size']):
            idx = order[st:st+c['batch_size']]
            bx = X[:, idx]
            if c['augment']:
                fill = -1.0 if c['preprocessing']=='minus_one_one' else 0.0
                bx = augment_batch(bx, rng, fill)
            by = y[idx]
            a, cache = net.forward(bx, training=True, return_cache=True)
            _, grads = net.backward(a, cache, by)
            for j,g in enumerate(grads):
                gmeans[j] += float(np.mean(np.abs(g['W'])))
                gnorms[j] += float(np.linalg.norm(g['W']))
            opt.step(grads, lr_at_epoch(c, epoch))
            batches += 1
        train = evaluate(net, X, y, c)
        val = evaluate(net, xv, yv, c)
        duration = time.perf_counter()-start_epoch
        row = {'epoch': epoch+1, 'train_loss':train['loss'], 'train_acc':train['accuracy'],
               'val_loss':val['loss'], 'val_acc':val['accuracy'], 'epoch_seconds':duration,
               'learning_rate':lr_at_epoch(c, epoch)}
        for j in range(len(net.layers)):
            row[f'grad_mean_L{j+1}'] = gmeans[j]/max(1,batches)
            row[f'grad_norm_L{j+1}'] = gnorms[j]/max(1,batches)
        rows.append(row)
        if not quiet:
            print(f"  {epoch+1:2d}/{c['epochs']} train={train['accuracy']:.3f} "
                  f"val={val['accuracy']:.3f} loss={val['loss']:.4f} "
                  f"time={duration:.1f}s", flush=True)
        if val['loss'] < best_val-1e-7:
            best_val = val['loss']
            bad_epochs = 0
            # Save weights, running BN statistics and optimizer-independent state.
            best_state = (copy.deepcopy(net.layers), copy.deepcopy(net.bn_running))
        else:
            bad_epochs += 1
            if c['early_stopping'] and bad_epochs >= c['patience']:
                if not quiet: print('  Early stopping triggered')
                break
    elapsed = time.perf_counter()-start_total
    if c['early_stopping'] and best_state is not None:
        net.layers, net.bn_running = best_state
    if run_dir is not None:
        pd.DataFrame(rows).to_csv(run_dir/'history.csv', index=False)
    return rows, elapsed


def slug(s):
    return ''.join(ch if ch.isalnum() or ch in ('_','-') else '_' for ch in str(s))[:60]


def json_dump(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_run(net, rows, run_dir, data):
    h = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    axes[0].plot(h.epoch, h.train_loss, label='Train')
    axes[0].plot(h.epoch, h.val_loss, label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss per epoch')
    axes[0].legend()
    axes[1].plot(h.epoch, h.train_acc, label='Train')
    axes[1].plot(h.epoch, h.val_acc, label='Validation')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy per epoch')
    axes[1].legend()
    save_fig(run_dir/'learning_curves.png')
    keys = [k for k in h if k.startswith('grad_mean_L')]
    plt.figure(figsize=(7,4))
    for k in keys:
        plt.plot(h.epoch, h[k], label=k)
    plt.xlabel('Epoch'); plt.ylabel('Mean |dW|')
    plt.title('Gradient magnitudes'); plt.legend(fontsize=8)
    save_fig(run_dir/'gradient_magnitudes.png')
    fig, axes = plt.subplots(1, len(net.layers), figsize=(3.5*len(net.layers),3.3))
    axes = np.array(axes).ravel()
    for i, ax in enumerate(axes):
        ax.hist(net.layers[i]['W'].ravel(), bins=45)
        ax.set_title(f'Weights: Layer {i+1}')
    save_fig(run_dir/'weight_histograms.png')
    sample = data['val'][0][:, :min(128, len(data['val'][1]))]
    _, cache = net.forward(sample, training=False, return_cache=True)
    plt.figure(figsize=(7,4))
    for i, ent in enumerate(cache['caches'][:-1]):
        plt.hist(ent['h'].ravel(), bins=50, alpha=.5, label=f'L{i+1}', density=True)
    if len(cache['caches'])>1:
        plt.legend(); plt.title('Hidden pre-activation distributions')
        save_fig(run_dir/'activation_histograms.png')
    else:
        plt.close()
    if net.layers[0]['W'].shape[1] == 784:
        w = net.layers[0]['W']
        fig, ax = plt.subplots(2, 5, figsize=(9,4))
        for i, a in enumerate(ax.flat):
            if i < w.shape[0]:
                a.imshow(w[i].reshape(28,28), cmap='coolwarm')
            a.axis('off')
        fig.suptitle('First-layer weights reshaped to 28 x 28')
        save_fig(run_dir/'first_layer_weights.png')


def make_case_config(overrides=None):
    c = copy.deepcopy(config)
    if overrides: c.update(copy.deepcopy(overrides))
    return c


def run_case(label, settings, data, group, args, test=True, quiet=False):
    c = make_case_config(settings)
    if args.epochs is not None: c['epochs'] = args.epochs
    if args.quick:
        c['epochs'] = min(c['epochs'], 2)
    validate(c)
    runid = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = OUTPUT_ROOT/group/(runid+'_'+slug(label))
    path.mkdir(parents=True, exist_ok=True)
    json_dump(path/'config.json', c)
    if group == 'preprocessing':
        raw = data['train'][0]
        json_dump(path/'raw_training_pixel_stats.json', {
            'mean_pixel_value': float(raw.mean()),
            'pixel_std': float(raw.std()),
            'pixels_with_zero_std': int(np.sum(raw.std(axis=1)==0))})
    subset = data
    if group == 'tuning':
        subset = subset_data(data, min(10000, len(data['train'][1])), None, None)
    processed = preprocess(subset, c['preprocessing'])
    X, y = processed['train']
    if X.shape[0] != c['layer_sizes'][0] or c['layer_sizes'][-1] != 10:
        raise ValueError('MNIST requires 784 inputs and 10 outputs')
    net = Network(c)
    init_loss = evaluate(net, processed['val'][0], processed['val'][1], c)['loss']
    print(f'[{group}] {label} | parameters={net.count_parameters():,} '
          f'| initial val loss={init_loss:.4f}', flush=True)
    rows, elapsed = training_time(net, processed, path, quiet=quiet)
    train = evaluate(net, *processed['train'], c)
    val = evaluate(net, *processed['val'], c)
    final_test = evaluate(net, *processed['test'], c) if test else None
    at90 = next((r['epoch'] for r in rows if r['val_acc']>=0.90), None)
    result = {'run_id':runid, 'experiment':group, 'label':label,
              'seed':c['seed'], 'parameters':net.count_parameters(),
              'epochs_run':len(rows), 'epochs_to_90_val_acc':at90,
              'train_acc':train['accuracy'], 'val_acc':val['accuracy'],
              'test_acc': final_test['accuracy'] if final_test else None,
              'train_loss':train['loss'], 'val_loss':val['loss'],
              'test_loss':final_test['loss'] if final_test else None,
              'overfit_gap':train['accuracy']-val['accuracy'],
              'training_seconds':elapsed, 'seconds_per_epoch':elapsed/len(rows),
              'init_val_loss':init_loss, 'config':json.dumps(c),
              'folder':str(path)}
    json_dump(path/'metrics.json', result)
    if not args.no_plots: plot_run(net, rows, path, processed)
    message = f"  FINAL val={val['accuracy']*100:.2f}% "
    if final_test is not None:
        message += f"test={final_test['accuracy']*100:.2f}% "
    print(message + f'time={elapsed:.1f}s | {path}', flush=True)
    return {'net':net, 'metrics':result, 'history':rows, 'data':processed, 'path':path}


def compare_and_report(group, results):
    directory = OUTPUT_ROOT/group
    directory.mkdir(parents=True, exist_ok=True)
    if not results: return
    rows = [r['metrics'] for r in results]
    pd.DataFrame(rows).to_csv(directory/'comparison.csv', index=False)
    if len(results) > 1:
        plt.figure(figsize=(max(8, len(results)*0.7), 4))
        vals = [r['metrics']['val_acc']*100 for r in results]
        plt.bar(range(len(results)), vals)
        plt.xticks(range(len(results)), [r['metrics']['label'] for r in results],
                   rotation=45, ha='right', fontsize=8)
        plt.ylabel('Validation accuracy (%)'); plt.title(group)
        save_fig(directory/'comparison_accuracy.png')
        plt.figure(figsize=(9,5))
        for r in results:
            h = r['history']
            plt.plot([row['epoch'] for row in h], [row['val_loss'] for row in h],
                     label=r['metrics']['label'])
        plt.xlabel('Epoch'); plt.ylabel('Validation loss')
        plt.title('Comparison: validation loss')
        plt.legend(fontsize=7, ncol=2)
        save_fig(directory/'comparison_loss.png')
    questions = QUESTIONS.get(group, ['What changed, and why?'])
    best = max(results, key=lambda r:r['metrics']['val_acc'])['metrics']
    md = [f'# {group.replace("_", " ").title()} experiment', '',
          '## Results', '', pd.DataFrame(rows)[['label','val_acc','test_acc',
            'training_seconds','overfit_gap']].to_markdown(index=False) if
          _has_tabulate() else 'See comparison.csv', '',
          f"Best observed validation accuracy: **{best['label']}** ({best['val_acc']:.2%}).",
          'This is a measurement, not a guarantee about other seeds or datasets.', '',
          '## Discussion questions (answer using graphs and results)', '']
    md += [f'{i}. {q}' for i,q in enumerate(questions,1)]
    md += ['', '## Key takeaways', '',
           '- Write what changed while other settings remained fixed.',
           '- Explain differences in validation loss, training time and overfitting.',
           '- Note any run-to-run variability; repeat important comparisons with more seeds.',
           '- For model selection use validation results; reserve the test set for a final check.', '']
    (directory/'DISCUSSION.md').write_text('\n'.join(md), encoding='utf-8')


def _has_tabulate():
    try: import tabulate; return True
    except ImportError: return False


def default_cases(mode):
    base = {'architecture':'plain','batch_norm':False, 'augment':False,
            'early_stopping':False, 'schedule':'constant'}
    def make(label, **kwargs): return (label, {**base, **kwargs})
    if mode == 'baseline':
        return [make('baseline')]
    if mode == 'activations':
        return [make(a, activations=[a,'softmax']) for a in HIDDEN]
    if mode == 'sigmoid_depth':
        return [make(f'{a}_5_hidden',layer_sizes=[784]+[128]*5+[10],
                     activations=[a]*5+['softmax']) for a in ('relu','sigmoid')]
    if mode == 'depth':
        return [make(f'{n}_hidden',layer_sizes=[784]+[128]*n+[10],
                     activations=['relu']*n+['softmax']) for n in (1,2,3,4,5,8,10)]
    if mode == 'width':
        return [make(f'{n}_neurons',layer_sizes=[784,n,n,10],
                     activations=['relu','relu','softmax']) for n in (32,64,128,256,512,1024)]
    if mode == 'learning_rate':
        return [make(f'lr_{x:g}',lr=x) for x in (.1,.05,.01,.005,.001,.0001,.00001)]
    if mode == 'lr_schedules':
        return [make(s,schedule=s) for s in ('constant','step','exponential','cosine')]
    if mode == 'batch_size':
        return [make(f'batch_{x}',batch_size=x) for x in (16,32,64,128,256,512,60000)]
    if mode == 'batch_scaling':
        return [make(f'batch_{b}_lr_{0.01*b/64:g}', batch_size=b,lr=0.01*b/64)
                for b in (32,64,128,256)]
    if mode == 'optimizers':
        return [make(f'{op}_lr_{lr}',optimizer=op,lr=lr)
                for op in OPTS for lr in (0.01, 0.001)]
    if mode == 'initialization':
        return [make(s,init=s,epochs=10,layer_sizes=[784,128,128,10],
                     activations=['relu','relu','softmax'])
                for s in ('zeros','small','large','xavier','he')]
    if mode == 'regularization':
        sz=[784,512,512,512,10]
        ac=['relu']*3+['softmax']
        return ([make('none',layer_sizes=sz,activations=ac)] +
                [make(f'weight_decay_{v:g}',layer_sizes=sz,activations=ac,
                      weight_decay=v) for v in (1e-4,1e-3,1e-2)] +
                [make(f'dropout_{v:g}',layer_sizes=sz,activations=ac,dropout=v)
                 for v in (.2,.5)] +
                [make('data_augmentation',layer_sizes=sz,activations=ac,augment=True),
                 make('early_stopping',layer_sizes=sz,activations=ac,early_stopping=True)])
    if mode == 'loss_functions':
        return [make('cross_entropy_softmax'),
                make('mse_softmax',loss='mse'),
                make('mse_sigmoid',loss='mse',activations=['relu','sigmoid'])]
    if mode == 'preprocessing':
        return [make(p,preprocessing=p) for p in ('raw','zero_one','minus_one_one','standardize')]
    if mode == 'residual':
        s=[784]+[128]*10+[10]; a=['relu']*10+['softmax']
        return [make('plain_10',layer_sizes=s,activations=a),
                make('residual_10',architecture='residual',layer_sizes=s,activations=a)]
    if mode == 'bottleneck':
        return [make('regular_512',layer_sizes=[784,512,512,512,10],
                     activations=['relu']*3+['softmax']),
                make('bottleneck_64',layer_sizes=[784,512,64,512,10],
                     activations=['relu']*3+['softmax'])]
    if mode == 'multi_branch':
        return [make('plain_2x128',layer_sizes=[784,256,10]),
                make('parallel_128_128',architecture='multi_branch',
                     branch_widths=[128,128])]
    if mode == 'output_activation':
        return [make('softmax_cross_entropy'),
                make('sigmoid_binary_cross_entropy',activations=['relu','sigmoid'])]
    if mode == 'batch_norm':
        sz=[784,128,128,128,10]
        return [make('without_batch_norm',layer_sizes=sz,activations=['relu']*3+['softmax']),
                make('with_batch_norm',batch_norm=True,layer_sizes=sz,
                     activations=['relu']*3+['softmax'])]
    return []


def plot_derivatives(path):
    xx = np.linspace(-5,5,600)
    plt.figure(figsize=(10,6))
    for activation in HIDDEN:
        plt.plot(xx, d_act(xx,activation), label=activation)
    plt.xlabel('Pre-activation z'); plt.ylabel('Activation derivative')
    plt.title('Activation derivatives and gradient flow'); plt.legend()
    save_fig(path/'activation_derivatives.png')


def gradient_check(path, seed=42):
    c = make_case_config({'layer_sizes':[4,3,2], 'activations':['tanh','softmax'],
                          'dropout':0.0,'weight_decay':0.0,'batch_norm':False})
    net = Network(c, dtype=np.float64)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(4,4))
    y = np.array([0,1,0,1])
    A, cache = net.forward(X, return_cache=True)
    _, grad = net.backward(A, cache, y)
    biggest=0.0
    eps=1e-5
    for i,p in enumerate(net.layers):
        for key in ('W','b'):
            for idx in np.ndindex(p[key].shape):
                old = p[key][idx]
                p[key][idx] = old+eps
                pos=calc_loss(net.forward(X), y, 'cross_entropy','softmax')
                p[key][idx] = old-eps
                neg=calc_loss(net.forward(X), y, 'cross_entropy','softmax')
                p[key][idx] = old
                numeric=(pos-neg)/(2*eps)
                analytic=grad[i][key][idx]
                error=abs(numeric-analytic)/max(1e-8,abs(numeric)+abs(analytic))
                biggest=max(biggest,error)
    print(f'Gradient check max relative error: {biggest:.3e} (target < 1e-6)')
    (path/'gradient_check.txt').write_text(
        f'Max relative error: {biggest:.12g}\nPass (1e-6): {biggest < 1e-6}\n',
        encoding='utf-8')
    return biggest


def run_debugging(data, args):
    path = OUTPUT_ROOT/'debugging'
    path.mkdir(parents=True, exist_ok=True)
    err = gradient_check(path)
    c = make_case_config({'lr':0.1,'epochs':1000, 'batch_size':10})
    X,y = preprocess(data, 'zero_one')['train']
    X=X[:,:10]; y=y[:10]
    net=Network(c)
    init = calc_loss(net.forward(X),y,'cross_entropy','softmax')
    opt=Optimizer(net)
    acc=0
    for i in range(1000):
        A, cache=net.forward(X,training=True,return_cache=True)
        _, grads=net.backward(A,cache,y)
        opt.step(grads,.1)
        if (i+1)%10 == 0:
            acc=(np.argmax(net.forward(X),axis=0)==y).mean()
            if acc==1.0: break
    print(f'Initial loss on 10 examples: {init:.4f}; after {i+1} updates: {acc*100:.1f}%')
    json_dump(path/'debug_results.json', {'gradient_relative_error':err,
              'initial_loss':init,'expected_initial_loss_ln10':math.log(10),
              'single_batch_accuracy':float(acc), 'updates':i+1})
    # Visualize learned digit-like first-layer filters.
    fig,axs=plt.subplots(2,5,figsize=(10,4))
    for j,ax in enumerate(axs.flat):
        ax.imshow(net.layers[0]['W'][j].reshape(28,28),cmap='coolwarm')
        ax.axis('off')
    save_fig(path/'overfit_first_layer_filters.png')
    (path/'DISCUSSION.md').write_text('# Debugging\n\n'+'\n'.join(QUESTIONS['debugging'])+
                                       '\n\nSee debug_results.json and filters.\n',encoding='utf-8')


def compute_roc(y, prob, positive):
    truth=(y==positive).astype(int)
    order=np.argsort(-prob, kind='mergesort')
    yy=truth[order]; p=np.cumsum(yy); f=np.cumsum(1-yy)
    P=max(1,truth.sum()); N=max(1,len(truth)-truth.sum())
    tpr=np.r_[0,p/P]; fpr=np.r_[0,f/N]
    return fpr,tpr,float(np.trapezoid(tpr,fpr))


def run_error_analysis(run):
    path=OUTPUT_ROOT/'error_analysis'
    path.mkdir(parents=True,exist_ok=True)
    net,c=run['net'],run['net'].cfg
    X,y=run['data']['test']
    res=evaluate(net,X,y,c,probabilities=True)
    pred,prob=res['predictions'],res['probabilities']
    matrix=np.zeros((10,10),dtype=int)
    np.add.at(matrix,(y,pred),1)
    pd.DataFrame(matrix,index=list(range(10)),columns=list(range(10))).to_csv(path/'confusion_matrix.csv')
    report=[]
    for k in range(10):
        tp=matrix[k,k]
        precision=tp/max(1,matrix[:,k].sum())
        recall=tp/max(1,matrix[k,:].sum())
        f1=2*precision*recall/max(1e-12,precision+recall)
        report.append({'digit':k,'precision':precision,'recall':recall,'f1':f1,
                       'support':int(matrix[k,:].sum())})
    pd.DataFrame(report).to_csv(path/'per_class_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(matrix)
    fig.colorbar(im,ax=ax); ax.set(xlabel='Predicted digit',ylabel='True digit',title='Confusion matrix')
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); save_fig(path/'confusion_matrix.png')
    wrong=np.where(pred!=y)[0][:25]
    if len(wrong):
        fig,axs=plt.subplots(5,5,figsize=(9,9))
        for ax in axs.flat: ax.axis('off')
        for ax,idx in zip(axs.flat,wrong):
            # Plot ORIGINAL image rather than normalized values when possible.
            xraw=(X[:,idx] if c['preprocessing']=='raw' else
                  X[:,idx]*255 if c['preprocessing']=='zero_one' else X[:,idx])
            ax.imshow(xraw.reshape(28,28),cmap='gray')
            ax.set_title(f'{y[idx]}→{pred[idx]}, {prob[idx,pred[idx]]:.2f}',fontsize=8)
        save_fig(path/'misclassified_examples.png')
    plt.figure(figsize=(7,6)); aucs={}
    for k in range(10):
        xroc,yroc,auc=compute_roc(y,prob[:,k],k)
        aucs[str(k)]=auc
        plt.plot(xroc,yroc,label=f'{k} AUC={auc:.2f}',lw=1.1)
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel('False positive rate');plt.ylabel('True positive rate')
    plt.title('One-vs-rest ROC curves');plt.legend(ncol=2,fontsize=8)
    save_fig(path/'roc_curves.png')
    pd.DataFrame({'true':y,'predicted':pred,'confidence':prob[np.arange(len(y)),pred]}).to_csv(
        path/'predictions.csv',index=False)
    json_dump(path/'metrics.json',{'accuracy':res['accuracy'],'loss':res['loss'],'roc_auc':aucs})
    (path/'DISCUSSION.md').write_text('# Error analysis\n\n'+'\n'.join(QUESTIONS['error_analysis'])+
                                       '\n\nUse the confusion matrix, ROC and wrong examples.\n',encoding='utf-8')
    print('Error analysis saved to:',path)


def run_mc_dropout(data,args):
    # Inverted dropout remains active at test time for multiple stochastic passes.
    r=run_case('dropout_0_2',{'dropout':.2,'epochs':15},data,'mc_dropout',args)
    X,y=r['data']['test']
    X=X[:,:min(500,len(y))];y=y[:X.shape[1]]
    samples=np.stack([r['net'].forward(X,mc_dropout=True).T for _ in range(20)])
    avg=samples.mean(axis=0)
    pred=np.argmax(avg,axis=1)
    uncertainty=np.mean(np.std(samples,axis=0),axis=1)
    records=pd.DataFrame({'true':y,'predicted':pred,'uncertainty':uncertainty,
                          'max_probability':avg.max(axis=1)})
    records.to_csv(OUTPUT_ROOT/'mc_dropout'/'uncertainties.csv',index=False)
    json_dump(OUTPUT_ROOT/'mc_dropout'/'summary.json',
              {'MC_accuracy_500':float(np.mean(pred==y)),
               'mean_prediction_std':float(uncertainty.mean()),'passes':20})
    compare_and_report('mc_dropout',[r])


def run_ensemble(data,args):
    group='ensemble'
    trained=[]
    for seed in (42,100,123,456,789):
        r=run_case(f'seed_{seed}',{'seed':seed},data,group,args,quiet=True)
        trained.append(r)
    X,y=trained[0]['data']['test']
    prob=np.zeros((10,len(y)),dtype=np.float64)
    for r in trained:
        net=r['net']
        for st in range(0,len(y),512):
            prob[:,st:st+512] += net.forward(X[:,st:st+512])/len(trained)
    ens=float(np.mean(prob.argmax(axis=0)==y))
    individual=[r['metrics']['test_acc'] for r in trained]
    json_dump(OUTPUT_ROOT/group/'ensemble_summary.json',
              {'ensemble_test_accuracy':ens,'individual_test_accuracies':individual,
               'mean_individual_accuracy':float(np.mean(individual))})
    compare_and_report(group,trained)
    print(f'Ensemble test accuracy: {ens:.2%} vs mean individual {np.mean(individual):.2%}')


def run_adversarial(data,args):
    r=run_case('fgsm_baseline',{'preprocessing':'zero_one'},data,'adversarial',args)
    net=r['net']; X,y=r['data']['test']
    # Generate adversarial examples without data leakage from the training set.
    epsilons=(0.0,0.05,0.1,0.2,0.3)
    results=[]
    for eps in epsilons:
        ok=0
        for st in range(0,len(y),128):
            bx=X[:,st:st+128];by=y[st:st+128]
            a,cache=net.forward(bx,return_cache=True)
            _,_,dx=net.backward(a,cache,by,input_gradient=True)
            adv=np.clip(bx+eps*np.sign(dx),0,1)
            ok += int(np.sum(net.forward(adv).argmax(axis=0)==by))
        results.append({'epsilon':eps,'test_accuracy':ok/len(y)})
    path=OUTPUT_ROOT/'adversarial'
    pd.DataFrame(results).to_csv(path/'fgsm_results.csv',index=False)
    plt.figure(figsize=(6,4))
    plt.plot([v['epsilon'] for v in results],[v['test_accuracy'] for v in results],marker='o')
    plt.xlabel('FGSM epsilon (pixel scale 0..1)');plt.ylabel('Test accuracy')
    plt.title('Accuracy under FGSM attack');save_fig(path/'fgsm_accuracy.png')
    compare_and_report('adversarial',[r])
    print('FGSM:',results)


def tuning(data,args):
    group='tuning'
    keys=['lr','batch_size','width','depth','dropout']
    values=[(0.1,.01,.001),(32,64,128),(64,128,256),(1,2,3),(0.,.2,.5)]
    combos=list(itertools.product(*values))
    rng=np.random.default_rng(42)
    if args.full_grid:
        grid=combos
    else:
        count=min(args.trials,len(combos))
        # A spaced grid subset (not a complete grid search).
        grid=[combos[int(i)] for i in np.linspace(0,len(combos)-1,count)]
    rand=[combos[int(i)] for i in rng.choice(len(combos),size=len(grid),replace=False)]
    gathered=[]
    for method, trials in [('grid' if args.full_grid else 'grid_subset',grid),('random',rand)]:
        for j, (lr,bs,width,depth,drop) in enumerate(trials):
            c={'lr':lr,'batch_size':bs,'layer_sizes':[784]+[width]*depth+[10],
               'activations':['relu']*depth+['softmax'], 'dropout':drop,
               'epochs':5}  # shorter runs for search, as PDF suggests 10k samples.
            r=run_case(f'{method}_{j+1}',c,data,group,args,test=False,quiet=True)
            r['metrics']['search_method']=method
            gathered.append(r)
    compare_and_report(group,gathered)
    best=max(gathered,key=lambda r:r['metrics']['val_acc'])
    # The test set is touched only once, for the selected validation winner.
    final=evaluate(best['net'],*best['data']['test'],best['net'].cfg)
    json_dump(OUTPUT_ROOT/group/'best_config.json',best['net'].cfg)
    json_dump(OUTPUT_ROOT/group/'best_validation_selection.json',
              {'winner_label':best['metrics']['label'],
               'winner_val_accuracy':best['metrics']['val_acc'],
               'winner_test_accuracy':final['accuracy'],
               'note':'Winner is selected by validation accuracy on the 10k-image training subset.'})
    print(f"Best tuning trial: {best['metrics']['label']} "
          f"validation={best['metrics']['val_acc']:.2%}, test={final['accuracy']:.2%}")


def main():
    parser=argparse.ArgumentParser(description='Complete NumPy MNIST ablation study runner')
    parser.add_argument('--run',choices=MODES,default='baseline')
    parser.add_argument('--list',action='store_true',help='Show available experiment names')
    parser.add_argument('--epochs',type=int,help='Override number of epochs for each run')
    parser.add_argument('--quick',action='store_true',help='Smoke test: 2 epochs, 500 train, 100 val/test')
    parser.add_argument('--train-limit',type=int,default=None)
    parser.add_argument('--val-limit',type=int,default=None)
    parser.add_argument('--test-limit',type=int,default=None)
    parser.add_argument('--trials',type=int,default=10,help='Grid subset / random trials each')
    parser.add_argument('--full-grid',action='store_true',help='Run all 243 grid combinations')
    parser.add_argument('--no-plots',action='store_true',help='Skip per-run plots (speed)')
    args=parser.parse_args()
    if args.list:
        print('Available experiments:\n  '+'\n  '.join(MODES))
        return
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.trials < 1: parser.error('--trials must be positive')
    data=load_data(config['seed'])
    data=subset_data(data,500 if args.quick else args.train_limit,
                     100 if args.quick else args.val_limit,
                     100 if args.quick else args.test_limit)
    OUTPUT_ROOT.mkdir(exist_ok=True)
    if args.run=='all':
        modes=[m for m in MODES if m not in ('all','baseline')]
        print('WARNING: All PDF experiments may take MANY HOURS on the full dataset.')
    else:
        modes=[args.run]
    for mode in modes:
        print('\n'+'='*15+' '+mode.upper()+' '+'='*15,flush=True)
        if mode=='tuning':
            tuning(data,args);continue
        if mode=='debugging':
            run_debugging(data,args);continue
        if mode=='mc_dropout':
            run_mc_dropout(data,args);continue
        if mode=='ensemble':
            run_ensemble(data,args);continue
        if mode=='adversarial':
            run_adversarial(data,args);continue
        if mode=='error_analysis':
            r=run_case('error_analysis_baseline',{},data,mode,args)
            run_error_analysis(r)
            compare_and_report(mode,[r]);continue
        cases=default_cases(mode)
        results=[]
        for label,settings in cases:
            results.append(run_case(label,settings,data,mode,args))
        compare_and_report(mode,results)
        if mode=='activations': plot_derivatives(OUTPUT_ROOT/mode)
    print('\nFinished. Outputs saved in:',OUTPUT_ROOT)


if __name__=='__main__':
    main()

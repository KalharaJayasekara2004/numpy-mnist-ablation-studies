import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots without opening windows.
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from pathlib import Path

lr = 0.00001

# Change the hyperparameters for your experiments here.
config = {
    'layer_sizes':  [784, 128, 10],
    'activations': ['relu', 'softmax'],  # One activation per layer after the input
    'init': 'he',
    'lr': lr,
    'batch_size': 64,
    'epochs': 20,
    'optimizer': 'sgd',
    'loss': 'cross_entropy',
    'weight_decay': 0.0,
    'dropout': 0.0,
    'seed': 42,
    'preprocessing': 'zero_one',  # raw, zero_one, minus_one_one, standardize
}

# Edit these two labels before each run to compare settings later.
experiment = 'learning_rate'
run_name = f'lr_{lr}'

# Read CSV files from the same folder as this script.
folder = Path(__file__).resolve().parent

def check_config(cfg):
    sizes = cfg['layer_sizes']
    acts = cfg['activations']
    if len(sizes) < 2 or any(n <= 0 for n in sizes):
        raise ValueError('layer_sizes must contain positive layer sizes.')
    if len(acts) != len(sizes) - 1:
        raise ValueError('Give one activation per hidden/output layer.')
    hidden = {'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'swish', 'linear'}
    if any(a not in hidden for a in acts[:-1]):
        raise ValueError('Unknown hidden-layer activation.')
    if acts[-1] not in ('softmax', 'sigmoid'):
        raise ValueError('Output activation must be softmax or sigmoid.')
    if cfg['optimizer'] not in ('sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam'):
        raise ValueError('Unknown optimizer.')
    if cfg['loss'] not in ('cross_entropy', 'mse'):
        raise ValueError('Loss must be cross_entropy or mse.')
    if not 0 <= cfg['dropout'] < 1:
        raise ValueError('dropout must be between 0 and 1 (exclusive of 1).')
    if cfg['weight_decay'] < 0 or cfg['lr'] <= 0 or cfg['batch_size'] <= 0 or cfg['epochs'] <= 0:
        raise ValueError('Check weight_decay, lr, batch_size, and epochs.')
    if cfg['preprocessing'] not in ('raw', 'zero_one', 'minus_one_one', 'standardize'):
        raise ValueError('Unknown preprocessing method.')

def load_data(filename):
    # Support both headerless MNIST CSVs and files with a header row.
    df = pd.read_csv(folder / filename, header=None, low_memory=False)
    if pd.isna(pd.to_numeric(df.iat[0, 0], errors='coerce')):
        df = df.iloc[1:]
    values = df.to_numpy(dtype=np.float64)
    return values[:, 1:].T, values[:, 0].astype(int)

def preprocess(X_train, X_val, X_test, method):
    if method == 'raw':
        return X_train, X_val, X_test
    if method == 'zero_one':
        return X_train / 255, X_val / 255, X_test / 255
    if method == 'minus_one_one':
        return X_train / 127.5 - 1, X_val / 127.5 - 1, X_test / 127.5 - 1
    mu = X_train.mean(axis=1, keepdims=True)
    sd = X_train.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1
    return (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd

def init_para(sizes, method, rng):
    W, b = [], []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        if method == 'he':
            scale = np.sqrt(2 / fan_in)
        elif method == 'xavier':
            scale = np.sqrt(2 / (fan_in + fan_out))
        elif method == 'small':
            scale = 0.01
        elif method == 'large':
            scale = 1.0
        elif method == 'zeros':
            scale = 0.0
        else:
            raise ValueError(f'Unknown initialization: {method}')
        W.append(rng.standard_normal((fan_out, fan_in)) * scale)
        b.append(np.zeros((fan_out, 1)))
    return W, b

def sigmoid(Z):
    A = np.empty_like(Z)
    positive = Z >= 0
    A[positive] = 1 / (1 + np.exp(-Z[positive]))
    e = np.exp(Z[~positive])
    A[~positive] = e / (1 + e)
    return A

def activation(Z, name):
    if name == 'relu':
        return np.maximum(0, Z)
    if name == 'sigmoid':
        return sigmoid(Z)
    if name == 'tanh':
        return np.tanh(Z)
    if name == 'leaky_relu':
        return np.where(Z > 0, Z, 0.01 * Z)
    if name == 'elu':
        return np.where(Z > 0, Z, np.expm1(np.minimum(Z, 0)))
    if name == 'swish':
        return Z * sigmoid(Z)
    if name == 'linear':
        return Z
    raise ValueError(f'Unknown activation: {name}')

def activation_derivative(Z, name):
    if name == 'relu':
        return (Z > 0).astype(float)
    if name == 'sigmoid':
        s = sigmoid(Z)
        return s * (1 - s)
    if name == 'tanh':
        return 1 - np.tanh(Z) ** 2
    if name == 'leaky_relu':
        return np.where(Z > 0, 1.0, 0.01)
    if name == 'elu':
        return np.where(Z > 0, 1.0, np.exp(np.minimum(Z, 0)))
    if name == 'swish':
        s = sigmoid(Z)
        return s + Z * s * (1 - s)
    if name == 'linear':
        return np.ones_like(Z)
    raise ValueError(f'Unknown activation: {name}')

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def one_hot(Y, classes):
    result = np.zeros((classes, Y.size))
    result[Y, np.arange(Y.size)] = 1
    return result

def get_predictions(A):
    return np.argmax(A, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def calculate_loss(A, Y, loss_name, output_act, with_gradient=False):
    y = one_hot(Y, A.shape[0])
    m = Y.size

    if loss_name == 'cross_entropy':
        if output_act == 'softmax':
            probs = np.clip(A[Y, np.arange(m)], 1e-12, 1.0)
            loss = -np.mean(np.log(probs))
        else:  # Sigmoid: binary cross-entropy for each class (one-vs-all).
            probs = np.clip(A, 1e-12, 1 - 1e-12)
            loss = -np.mean(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs), axis=0))
        dZ = (A - y) / m

    else:  # Mean squared error: 1/2 * mean sum of squared differences.
        diff = A - y
        loss = 0.5 * np.mean(np.sum(diff ** 2, axis=0))
        dA = diff / m
        if output_act == 'softmax':
            dZ = A * (dA - np.sum(A * dA, axis=0, keepdims=True))
        else:  # Sigmoid derivative.
            dZ = dA * A * (1 - A)

    return (loss, dZ) if with_gradient else loss

def forward_prop(W, b, X, acts, dropout=0.0, rng=None):
    A = [X]
    Z, masks = [], []

    for i in range(len(W)):
        z = W[i] @ A[-1] + b[i]
        a = softmax(z) if acts[i] == 'softmax' else activation(z, acts[i])
        mask = None

        # Inverted dropout: only on hidden layers and only during training.
        if i < len(W) - 1 and dropout > 0 and rng is not None:
            mask = (rng.random(a.shape) >= dropout) / (1 - dropout)
            a = a * mask

        Z.append(z)
        A.append(a)
        masks.append(mask)

    return Z, A, masks

def backward_prop(Z, A, W, Y, cfg, masks):
    loss, dZ = calculate_loss(A[-1], Y, cfg['loss'], cfg['activations'][-1], with_gradient=True)
    dW = [None] * len(W)
    db = [None] * len(W)

    for i in range(len(W) - 1, -1, -1):
        # L2 weight decay affects weights, not biases.
        dW[i] = dZ @ A[i].T + cfg['weight_decay'] * W[i]
        db[i] = np.sum(dZ, axis=1, keepdims=True)

        if i > 0:
            dA = W[i].T @ dZ
            if masks[i - 1] is not None:
                dA *= masks[i - 1]
            dZ = dA * activation_derivative(Z[i - 1], cfg['activations'][i - 1])

    # Loss displayed during training includes the L2 penalty.
    loss += 0.5 * cfg['weight_decay'] * sum(np.sum(w ** 2) for w in W)
    return loss, dW, db

def init_optimizer_state(W, b):
    # State for SGD with momentum, AdaGrad, RMSProp, and Adam.
    parameters = W + b
    return {
        'step': 0,
        'velocity': [np.zeros_like(p) for p in parameters],
        'avg_sq': [np.zeros_like(p) for p in parameters],
        'avg_grad': [np.zeros_like(p) for p in parameters],
    }

def update_params(W, b, dW, db, cfg, state):
    opt = cfg['optimizer']
    lr = cfg['lr']
    state['step'] += 1
    step = state['step']
    momentum, beta1, beta2, eps = 0.9, 0.9, 0.999, 1e-8

    for i, (p, g) in enumerate(zip(W + b, dW + db)):
        if opt == 'sgd':
            update = g
        elif opt in ('momentum', 'nesterov'):
            v = state['velocity'][i]
            v *= momentum
            v += g
            update = v  # For Nesterov, gradient is evaluated at lookahead weights below.
        elif opt == 'adagrad':
            v = state['avg_sq'][i]
            v += g ** 2
            update = g / (np.sqrt(v) + eps)
        elif opt == 'rmsprop':
            v = state['avg_sq'][i]
            v *= 0.9
            v += 0.1 * g ** 2
            update = g / (np.sqrt(v) + eps)
        elif opt == 'adam':
            avg = state['avg_grad'][i]
            sq = state['avg_sq'][i]
            avg *= beta1
            avg += (1 - beta1) * g
            sq *= beta2
            sq += (1 - beta2) * g ** 2
            avg_corrected = avg / (1 - beta1 ** step)
            sq_corrected = sq / (1 - beta2 ** step)
            update = avg_corrected / (np.sqrt(sq_corrected) + eps)
        p -= lr * update
    return W, b

# ============== Training, reporting, and plotting ==============

def evaluate(W, b, X, Y, cfg, detailed=False):
    # Evaluate in chunks to avoid huge arrays when testing wide/deep networks.
    loss_sum, correct = 0.0, 0
    preds, probs = [], []
    for start in range(0, Y.size, 512):
        y = Y[start:start + 512]
        _, A, _ = forward_prop(W, b, X[:, start:start + 512], cfg['activations'])
        p = A[-1]
        loss_sum += calculate_loss(p, y, cfg['loss'], cfg['activations'][-1]) * y.size
        labels = get_predictions(p)
        correct += np.count_nonzero(labels == y)
        if detailed:
            preds.append(labels)
            probs.append(p)
    return (loss_sum / Y.size, correct / Y.size,
            np.concatenate(preds) if detailed else None,
            np.hstack(probs) if detailed else None)

def gradient_descent(X, Y, X_val, Y_val, cfg):
    rng = np.random.default_rng(cfg['seed'])
    W, b = init_para(cfg['layer_sizes'], cfg['init'], rng)
    state = init_optimizer_state(W, b)
    history = []
    for epoch in range(cfg['epochs']):
        start_time = time.perf_counter()
        indices = rng.permutation(Y.size)
        grad_sum = np.zeros(len(W))
        grad_l2 = np.zeros(len(W))
        batches = 0
        for start in range(0, Y.size, cfg['batch_size']):
            ids = indices[start:start + cfg['batch_size']]
            X_batch, Y_batch = X[:, ids], Y[ids]
            # True Nesterov: compute gradient at the lookahead parameters.
            if cfg['optimizer'] == 'nesterov':
                for p, v in zip(W + b, state['velocity']):
                    p -= cfg['lr'] * 0.9 * v
            Z, A, masks = forward_prop(W, b, X_batch, cfg['activations'], cfg['dropout'], rng)
            _, dW, db = backward_prop(Z, A, W, Y_batch, cfg, masks)
            if cfg['optimizer'] == 'nesterov':
                for p, v in zip(W + b, state['velocity']):
                    p += cfg['lr'] * 0.9 * v
            grad_sum += [np.mean(np.abs(g)) for g in dW]
            grad_l2 += [np.linalg.norm(g) for g in dW]
            batches += 1
            W, b = update_params(W, b, dW, db, cfg, state)

        train_loss, train_acc, _, _ = evaluate(W, b, X, Y, cfg)
        val_loss, val_acc, _, _ = evaluate(W, b, X_val, Y_val, cfg)
        row = dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                   train_acc=train_acc, val_acc=val_acc,
                   epoch_seconds=time.perf_counter() - start_time)
        for i in range(len(W)):
            row[f'grad_mean_L{i+1}'] = grad_sum[i] / batches
            row[f'grad_norm_L{i+1}'] = grad_l2[i] / batches
        history.append(row)
        print(f"Epoch {epoch+1:>3}/{cfg['epochs']} | Train: {train_acc:.2%} "
              f"Val: {val_acc:.2%} | Loss: {train_loss:.4f}/{val_loss:.4f} "
              f"| {row['epoch_seconds']:.1f}s")
    return W, b, pd.DataFrame(history)

def plot_curves(history, output):
    for metric, ylabel, filename in [('loss', 'Cross-entropy / MSE loss', 'loss_curves.png'),
                                    ('acc', 'Accuracy', 'accuracy_curves.png')]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(history.epoch, history['train_' + metric], label='Train')
        plt.plot(history.epoch, history['val_' + metric], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title('Training vs validation ' + metric)
        plt.legend()
        plt.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig(output / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(7, 4.5))
    for col in history.columns:
        if col.startswith('grad_norm_'):
            plt.plot(history.epoch, history[col], label=col.replace('grad_norm_', 'Layer '))
    plt.xlabel('Epoch')
    plt.ylabel('Mean per-batch gradient L2 norm')
    plt.title('Gradient magnitude across layers')
    plt.legend(fontsize=8)
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(output / 'gradient_norms.png', dpi=150)
    plt.close()

def plot_weights(W, b, X_val, cfg, output):
    plt.figure(figsize=(7, 4.5))
    for i, w in enumerate(W):
        plt.hist(w.ravel(), bins=60, alpha=.45, label=f'Layer {i+1}')
    plt.title('Final weight distributions')
    plt.xlabel('Weight value')
    plt.ylabel('Count')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / 'weight_histograms.png', dpi=150)
    plt.close()

    Z, A, _ = forward_prop(W, b, X_val[:, :min(100, X_val.shape[1])], cfg['activations'])
    plt.figure(figsize=(7, 4.5))
    for i, a in enumerate(A[1:-1]):
        plt.hist(a.ravel(), bins=60, alpha=.4, label=f'Hidden layer {i+1}')
    if len(A) > 2:
        plt.title('Hidden-layer activation distributions')
        plt.legend(fontsize=8)
        plt.xlabel('Activation')
        plt.tight_layout()
        plt.savefig(output / 'activation_histograms.png', dpi=150)
    plt.close()

    # Plot the chosen hidden activation and its derivative.
    if len(cfg['activations']) > 1:
        z = np.linspace(-5, 5, 500)
        act = cfg['activations'][0]
        plt.figure(figsize=(7, 4.5))
        plt.plot(z, activation(z, act), label=act)
        plt.plot(z, activation_derivative(z, act), label='Derivative')
        plt.title(f'Activation function and derivative: {act}')
        plt.xlabel('z')
        plt.legend()
        plt.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig(output / 'activation_derivative.png', dpi=150)
        plt.close()

def plot_errors(X_test, Y_test, preds, probs, output):
    classes = probs.shape[0]
    cm = np.zeros((classes, classes), dtype=int)
    np.add.at(cm, (Y_test, preds), 1)
    pd.DataFrame(cm, index=range(classes), columns=range(classes)).to_csv(output / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(classes), yticks=range(classes), xlabel='Predicted',
           ylabel='True', title='Test confusion matrix')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=150)
    plt.close(fig)

    report = []
    for c in range(classes):
        tp = cm[c, c]
        precision = tp / max(cm[:, c].sum(), 1)
        recall = tp / max(cm[c].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        report.append(dict(digit=c, precision=precision, recall=recall,
                           f1=f1, support=cm[c].sum()))
    pd.DataFrame(report).to_csv(output / 'classification_report.csv', index=False)

    wrong = np.flatnonzero(Y_test != preds)[:12]
    if X_test.shape[0] == 784 and len(wrong):
        fig, axes = plt.subplots(3, 4, figsize=(9, 7))
        for ax in axes.ravel():
            ax.axis('off')
        for ax, j in zip(axes.ravel(), wrong):
            ax.imshow(X_test[:, j].reshape(28, 28), cmap='gray')
            ax.set_title(f'True {Y_test[j]}, pred {preds[j]} ({probs[preds[j], j]:.2f})', fontsize=9)
        fig.suptitle('Misclassified test images (predicted score in parentheses)')
        fig.tight_layout()
        fig.savefig(output / 'misclassified_images.png', dpi=150)
        plt.close(fig)

    # One-versus-rest ROC curves, without requiring scikit-learn.
    fig, ax = plt.subplots(figsize=(7, 5))
    roc_rows = []
    for c in range(classes):
        labels = (Y_test == c).astype(int)
        pos = labels.sum()
        neg = labels.size - pos
        if pos == 0 or neg == 0:
            continue
        order = np.argsort(-probs[c], kind='stable')
        tp = np.r_[0, np.cumsum(labels[order])]
        fp = np.r_[0, np.cumsum(1 - labels[order])]
        tpr, fpr = tp / pos, fp / neg
        auc = np.trapezoid(tpr, fpr) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr)
        roc_rows.append(dict(digit=c, auc=auc))
        ax.plot(fpr, tpr, label=f'{c}: AUC {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC curves (one-vs-rest)')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'roc_curves.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(roc_rows).to_csv(output / 'roc_auc.csv', index=False)

def update_comparison(results_folder, group):
    # Compare past runs from the same manually named experiment.
    files = sorted(results_folder.glob(f'{group}_*/history.csv'))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for file in files[-15:]:
        df = pd.read_csv(file)
        name = file.parent.name[len(group) + 1:].rsplit('_', 1)[0]
        axes[0].plot(df.epoch, df.val_acc, label=name)
        axes[1].plot(df.epoch, df.val_loss, label=name)
    for ax, title, ylabel in zip(axes, ['Validation accuracy', 'Validation loss'], ['Accuracy', 'Loss']):
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_folder / f'compare_{group}.png', dpi=150)
    plt.close(fig)

def main():
    check_config(config)
    print('Loading training and test data...')
    X, Y = load_data('mnist_train.csv')
    X_test, Y_test = load_data('mnist_test.csv')
    if X.shape[0] != config['layer_sizes'][0] or X_test.shape[0] != X.shape[0]:
        raise ValueError('Input layer size must match the number of image pixels.')
    if Y.min() < 0 or Y.max() >= config['layer_sizes'][-1]:
        raise ValueError('Output layer size does not match the labels.')

    # Same deterministic split for all runs using the same seed.
    rng = np.random.default_rng(config['seed'])
    ids = rng.permutation(Y.size)
    n_val = min(10000, max(1, round(Y.size / 6)))
    if Y.size - n_val < 1:
        raise ValueError('Need at least 2 training images to make a validation split.')
    train_ids, val_ids = ids[n_val:], ids[:n_val]
    X_train, Y_train = X[:, train_ids], Y[train_ids]
    X_val, Y_val = X[:, val_ids], Y[val_ids]
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test, config['preprocessing'])
    del X, Y

    results_folder = folder / 'Results'
    results_folder.mkdir(exist_ok=True)
    safe = lambda s: ''.join(c if c.isalnum() or c in '-_' else '_' for c in s)
    group = safe(experiment)
    label = safe(run_name)
    run_id = f'{group}_{label}_{datetime.now():%Y%m%d-%H%M%S-%f}'
    output = results_folder / run_id
    output.mkdir()
    with open(output / 'config.json', 'w', encoding='utf-8') as file:
        json.dump(config | {'experiment': experiment, 'run_name': run_name,
                            'train_images': Y_train.size, 'val_images': Y_val.size,
                            'test_images': Y_test.size}, file, indent=2)

    print('Config:', config)
    print(f'Data sizes: train={Y_train.size}, val={Y_val.size}, test={Y_test.size}')
    start = time.perf_counter()
    W, b, history = gradient_descent(X_train, Y_train, X_val, Y_val, config)
    train_time = time.perf_counter() - start
    test_loss, test_acc, predictions, probs = evaluate(W, b, X_test, Y_test, config, detailed=True)
    print(f'Finished. Test accuracy: {test_acc:.2%}, test loss: {test_loss:.4f}')

    history.to_csv(output / 'history.csv', index=False)
    reaching = history.loc[history.val_acc >= .9, 'epoch']
    summary = dict(run_id=run_id, experiment=experiment, run_name=run_name,
                   test_acc=test_acc, test_loss=test_loss,
                   train_acc=history.train_acc.iloc[-1], val_acc=history.val_acc.iloc[-1],
                   overfit_gap=history.train_acc.iloc[-1] - history.val_acc.iloc[-1],
                   epochs_to_90=int(reaching.iloc[0]) if len(reaching) else None,
                   parameters=sum(w.size + bias.size for w, bias in zip(W, b)),
                   training_seconds=train_time, seconds_per_epoch=train_time / len(history),
                   **{key: str(value) for key, value in config.items()})
    pd.DataFrame([summary]).to_csv(output / 'summary.csv', index=False)
    log_file = results_folder / 'all_experiments.csv'
    pd.DataFrame([summary]).to_csv(log_file, mode='a', index=False, header=not log_file.exists())

    plot_curves(history, output)
    plot_weights(W, b, X_val, config, output)
    plot_errors(X_test, Y_test, predictions, probs, output)
    (output / 'discussion_notes.md').write_text(
        '# Experiment: ' + experiment + ' / ' + run_name + '\n\n'
        'What single setting was changed?\n\n'
        'What happened to loss, accuracy, gradient size and training time?\n\n'
        'Why did these changes occur?\n\n'
        'What did you learn? What would you test next?\n', encoding='utf-8')
    update_comparison(results_folder, group)
    print('Saved plots, results, and tables to:', output)
    print('Comparison plot:', results_folder / f'compare_{group}.png')

if __name__ == '__main__':
    main()

import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from ad import utils


CMAP1 = 'RdGy_r'
CMAP2 = 'gist_heat_r'
CMAP3 = 'Greys_r'
DEFAULT_CMAP = None


def set_style(default_cmap=CMAP2, **kwargs):
    # further customization
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 20
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.alpha'] = 0.65
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['figure.figsize'] = (12, 10)
    mpl.rcParams['figure.autolayout'] = False

    for k, v in kwargs.items():
        mpl.rcParams[k] = v

    if default_cmap is not None:
        set_colormap(cmap=default_cmap)


def set_colormap(cmap):
    global DEFAULT_CMAP
    DEFAULT_CMAP = cmap


def sort_legend(ax, by_value: list, reverse=False) -> tuple:
    """Based on:  https://stackoverflow.com/a/27512450"""
    handles, labels = ax.get_legend_handles_labels()
    _, handles, labels = zip(*sorted(zip(by_value, handles, labels),
                                     key=lambda t: t[0], reverse=bool(reverse)))
    return handles, labels


def get_colormap(which='viridis', bkg_color='white', levels=1024):
    return cm.get_cmap(which, levels).with_extremes(under=bkg_color)


def heatmap(arr: np.array, size=(12, 10), ax=None, show=True, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    sns.heatmap(arr, **kwargs, ax=ax)

    if show:
        plt.show()

    return ax


def track(image: np.ndarray, size=(12, 12), ax=None, save=None, path='plot', **kwargs):
    track = tf.unstack(image, axis=-1)
    track = np.concatenate(track, axis=1)

    ax = heatmap(track, size=size, ax=ax, **kwargs)

    ax.set_title('Trk - ECAL - HCAL', fontsize=15)
    ax.set_ylabel(r'$\phi$ cell', fontsize=15)
    ax.set_xlabel(r'$\eta$ cell', fontsize=15)

    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')

    plt.show()


def plot_track(x: np.ndarray, border_width=0, **kwargs):
    ax = heatmap(np.sum(x, axis=-1), **kwargs)

    if border_width > 0:
        # https://www.geeksforgeeks.org/how-to-add-a-frame-to-a-seaborn-heatmap-figure-in-python/
        ax.axhline(y=0, color='k', linewidth=border_width)
        ax.axhline(y=x.shape[0], color='k', linewidth=border_width)
        ax.axvline(x=0, color='k', linewidth=border_width)
        ax.axvline(x=x.shape[1], color='k', linewidth=border_width)


def compare(*args, cmap=None, save: str = None, title: list = None, show_energy=True,
            path='plot', v_min=0.0, v_max=0.1, fonts: dict = None, **kwargs):
    assert len(args) > 0

    cmap = cmap or DEFAULT_CMAP
    fonts = fonts or {}
    axes = utils.get_plot_axes(rows=1, cols=len(args))

    if len(args) == 1:
        axes = [axes]

    if title is None:
        title = ['Ground-truth']

        if len(args) >= 2:
            title.append('Reconstruction')

        if len(args) >= 3:
            title.append('Diff: |GT - Pred|')

        assert len(args) == len(title)
    else:
        assert isinstance(title, list)
        assert len(args) == len(title)

    if isinstance(args[0], list) or len(args[0].shape) == 4:
        i = np.random.choice(len(args[0]))
        print(f'i: {i}')

        args = [x[i] for x in args]
    else:
        i = 0

    for x, ax, text in zip(args, axes, title):
        if v_max == 'none':
            v_max = np.max(x)

        if v_min == 'none':
            v_min = np.min(x)

        if cmap.lower() == 'rgb':
            ax.imshow(x, **kwargs)
        else:
            plot_track(x, cmap=cmap, ax=ax, vmin=v_min, vmax=v_max, show=False, **kwargs)

        e_t_max = np.max(x).item()
        e_t_tot = np.sum(x).item()

        if not show_energy:
            ax.set_title(text, fontsize=fonts.get('title', None))
        else:
            ax.set_title(f'{text}\n max E = {round(e_t_max, 2)}; total E = {round(e_t_tot, 2)}',
                         fontsize=fonts.get('title', None))

        ax.set_xlabel(r'$\eta$ cell', fontsize=fonts.get('axis', None))
        ax.set_ylabel(r'$\phi$ cell', fontsize=fonts.get('axis', None))

    plt.tight_layout()

    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}-{i}.png'), bbox_inches='tight')

    plt.show()


def compare_channels(*images, cmap='RdGy_r', v_min=None, v_max=None, **kwargs):
    text = ['Trk', 'ECAL', 'HCAL']

    if isinstance(v_min, (int, float)) or (v_min is None):
        v_min = [v_min] * images[0].shape[-1]
    else:
        assert len(v_min) == images[0].shape[-1]

    if isinstance(v_max, (int, float)) or (v_max is None):
        v_max = [v_max] * images[0].shape[-1]
    else:
        assert len(v_max) == images[0].shape[-1]

    for h in images:
        axes = utils.get_plot_axes(rows=1, cols=3)

        for c, (title, ax) in enumerate(zip(text, axes)):
            ax = heatmap(h[..., c], ax=ax, show=False, cmap=cmap, vmin=v_min[c], vmax=v_max[c], **kwargs)
            ax.set_title(title)

        plt.show()


def energy_histogram(data: dict, x_label: str, ax=None, bins=100, show=True, size=(12, 10),
                     var_range=None, log_scale=False, hatch_fn=None, legend='upper right'):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    if hatch_fn is None:
        hatch_fn = lambda key: 'QCD' in key
    else:
        assert callable(hatch_fn)

    if not isinstance(var_range, tuple):
        x_min = np.inf
        x_max = -np.inf

        for value in data.values():
            if isinstance(value, tuple):
                v, _, _ = value  # (true_e - pred_e, mean, std)
            else:
                v = value

            x_min = min(x_min, v.min())
            x_max = max(x_max, v.max())

        var_range = (x_min, x_max)

    for k, value in data.items():
        if isinstance(value, tuple):
            v, mu, std = value
            label = f'{k}: {round(mu, 2)} ({round(std, 2)})'
        else:
            v = value
            label = k

        ax.hist(v, bins=bins, label=label, range=var_range, density=True,
                histtype='step', hatch='//' if hatch_fn(k) else None)

    ax.set_xlabel(str(x_label))
    ax.set_ylabel('Probability')
    ax.legend(loc=str(legend))

    if log_scale:
        ax.set_yscale('log')

    if show:
        plt.tight_layout()
        plt.show()


# -------------------------------------------------------------------------------------------------
# -- ROCs
# -------------------------------------------------------------------------------------------------

def roc_losses(qcd_losses: dict, suep_losses: dict, scale='log', bins=100, x_limits: dict = None):
    """Plots a ROC curve using various losses as discriminator"""
    from sklearn.metrics import roc_auc_score, roc_curve

    if not isinstance(x_limits, dict):
        x_limits = {}

    for k, qcd_loss in qcd_losses.items():
        suep_loss = [losses[k] for _, losses in suep_losses.items()]
        suep_loss = np.concatenate(suep_loss)

        # compute roc
        y_true = np.concatenate([
            np.zeros_like(qcd_loss), np.ones_like(suep_loss)])

        y_score = np.concatenate([qcd_loss, suep_loss])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        # plot
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)
        loss_range = (min(qcd_loss.min(), suep_loss.min()),
                      max(qcd_loss.max(), suep_loss.max()))

        # histogram
        ax1.hist(qcd_loss, bins=bins, range=loss_range, density=True,
                 label='QCD', histtype='step')

        ax1.hist(suep_loss, bins=bins, range=loss_range, density=True,
                 label='SUEP', histtype='step', hatch='//')

        ax1.set_xlabel(k)
        ax1.set_ylabel('Probability')
        ax1.legend(loc='upper right')

        # set x limit for histogram
        left, right = ax1.get_xlim()
        right = x_limits.get(k, right)
        ax1.set_xlim(left, right)

        # ROC
        ax2.plot(fpr, tpr, label=f'{k}, AUC = {round(auc * 100, 2)}%')

        ax2.set_xlabel('Background efficiency (FPR)')
        ax2.set_yscale(scale)
        ax2.set_ylabel('Signal efficiency (TPR)')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()


def roc_per_mass(bkg_scores: dict, signal_scores: dict, scale='linear', bins=100, x_limits: dict = None,
                 legend_hist='upper right', legend_roc='lower right', fontsize=18):
    """Plots a ROC curve using various losses as discriminator"""
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = dict()

    if not isinstance(x_limits, dict):
        x_limits = {}

    for k, bkg_score in bkg_scores.items():
        other_score = [scores[k] for _, scores in signal_scores.items()]
        other_score = np.concatenate(other_score)

        # plot
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)
        loss_range = (min(bkg_score.min(), other_score.min()),
                      max(bkg_score.max(), other_score.max()))

        limits = x_limits.get(k, float(loss_range[1]))

        if isinstance(limits, tuple):
            inf, sup = limits
            loss_range = (max(inf, loss_range[0]), min(sup, loss_range[1]))
        else:
            assert isinstance(limits, (int, float))
            loss_range = (loss_range[0], min(limits, loss_range[1]))

        # histogram
        ax1.hist(bkg_score, bins=bins, range=loss_range, density=True,
                 label=utils.get_bkg_name(), histtype='step', hatch='//')

        curves[k] = {}

        for h, other_score in signal_scores.items():
            sig_score = other_score[k]
            label = f'{utils.get_name_from(mass=h)} ({h})'

            # compute roc
            y_true = np.concatenate([
                np.zeros_like(bkg_score), np.ones_like(sig_score)])

            y_score = np.concatenate([bkg_score, sig_score])

            fpr, tpr, t = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            curves[k][label] = dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=t)

            ax1.hist(sig_score, bins=bins, range=loss_range, density=True,
                     label=label, histtype='step')
            # ROC
            ax2.plot(fpr, tpr, label=f'{k} ({h}), AUC = {round(auc * 100, 2)}%')

        ax1.set_xlabel(k)
        ax1.set_ylabel('Probability', fontsize=fontsize)
        ax1.legend(loc=str(legend_hist), fontsize=fontsize - 4)

        # # set x limit for histogram
        # left, right = ax1.get_xlim()
        # right = x_limits.get(k, right)
        # ax1.set_xlim(left, right)

        ax2.set_xlabel('Background efficiency (FPR)', fontsize=fontsize)
        ax2.set_yscale(scale)
        ax2.set_ylabel('Signal efficiency (TPR)', fontsize=fontsize)
        ax2.legend(loc=str(legend_roc), fontsize=fontsize - 2)

        plt.tight_layout()
        plt.show()

    return curves


def roc_per_mass_stacked(bkg_scores: dict, signal_scores: dict, scale='linear', bins=100, fontsize=18,
                         legend_hist='upper right', legend_roc='lower right', weight=False,
                         thresholds: dict = None) -> dict:
    """Plots a ROC curve using various losses as discriminator"""
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = dict()
    thresholds = thresholds or {}

    for k, qcd_loss in bkg_scores.items():
        score = [scores[k] for _, scores in signal_scores.items()]
        score = np.concatenate(score)

        # plot
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)
        loss_range = (min(qcd_loss.min(), score.min()),
                      max(qcd_loss.max(), score.max()))

        # histogram
        ax1.hist(qcd_loss, bins=bins, range=loss_range, density=True,
                 label=utils.get_bkg_name(), histtype='step')

        scores = []
        labels = []
        curves[k] = {}

        for h, score in signal_scores.items():
            s_loss = score[k]
            key = f'{utils.get_name_from(mass=h)} ({h})'

            scores.append(s_loss)
            labels.append(key)

            # compute roc
            y_true = np.concatenate([
                np.zeros_like(qcd_loss), np.ones_like(s_loss)])

            if weight:
                w_suep = len(qcd_loss) / len(s_loss)
                w = np.concatenate([
                    np.ones_like(qcd_loss), w_suep * np.ones_like(s_loss)])
            else:
                w = None

            y_score = np.concatenate([qcd_loss, s_loss])

            fpr, tpr, t = roc_curve(y_true, y_score, sample_weight=w)
            auc = roc_auc_score(y_true, y_score)

            # ROC
            ax2.plot(fpr, tpr, label=f'{k} ({h}), AUC = {round(auc * 100, 2)}%')

            if key in thresholds:
                # find index
                idx = np.abs(t - thresholds[key]).argmin()
                ax2.scatter(fpr[idx], tpr[idx])

            curves[k][key] = dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=t)

        ax1.hist(scores, bins=bins, range=loss_range, density=True,
                 label=labels, stacked=True, histtype='step', hatch='//')

        ax1.set_xlabel(k, fontsize=fontsize)
        ax1.set_ylabel('Probability', fontsize=fontsize)
        ax1.legend(loc=str(legend_hist), fontsize=fontsize - 4)

        ax2.set_xlabel('Background efficiency (FPR)', fontsize=fontsize)
        ax2.set_ylabel('Signal efficiency (TPR)', fontsize=fontsize)
        ax2.set_yscale(scale)
        ax2.legend(loc=str(legend_roc), fontsize=fontsize - 2)

        plt.tight_layout()
        plt.show()

    return curves


# -------------------------------------------------------------------------------------------------

def pixels_hist(*counts, labels: list, bins=100, size=(12, 10), legend='best',
                ax=None, show=True, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    ax.hist(counts, bins=bins, label=labels, histtype='step', density=True, **kwargs)
    ax.set_xlabel('# Non-zero pixels')
    ax.set_ylabel('Frac. of Images')

    ax.legend(loc=str(legend))

    if show:
        plt.show()


def pixels_hist_stacked(qcd_counts, suep_counts: dict, bins=100, size=(12, 10),
                        legend='best', ax=None, show=True, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    x_range = [qcd_counts.min(), qcd_counts.max()]

    for v in suep_counts.values():
        x_range = [min(v.min(), x_range[0]),
                   max(v.max(), x_range[1])]

    ax.hist(qcd_counts, bins=bins, range=x_range, label='QCD',
            histtype='step', density=True, **kwargs)

    ax.hist(list(suep_counts.values()), bins=bins, range=x_range, density=True,
            label=list(suep_counts.keys()), histtype='step', hatch='//', stacked=True)

    ax.set_xlabel('# Non-zero pixels')
    ax.set_ylabel('Frac. of Images')

    ax.legend(loc=str(legend))

    if show:
        plt.show()


def history(h, keys: list, rows=2, cols=2, size=8):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    axes = np.reshape(axes, newshape=[-1])

    for ax, k in zip(axes, keys):
        ax.plot(h.epoch, h.history[k], marker='o', markersize=10, label='train')

        if f'val_{k}' in h.history:
            ax.plot(h.epoch, h.history[f'val_{k}'], marker='o', markersize=10, label='valid')

        ax.set_xlabel('# Epoch', fontsize=20)
        ax.set_ylabel(k.upper(), rotation="vertical", fontsize=20)

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        ax.grid(alpha=0.5, linestyle='dashed')
        ax.legend(loc='best')

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------------------------
# -- Latent space
# -------------------------------------------------------------------------------------------------

# TODO: rename to "latents"
def latent(x, y, title: str, ax=None, size=(12, 10), show=True):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

    for i, label in enumerate(np.unique(y)):
        x_ = x[y == label]
        ax.scatter(x_[:, 0], x_[:, 1], s=10, label=utils.get_name(label), zorder=-i)

    ax.set_xlabel(r'$z_0$')
    ax.set_ylabel(r'$z_1$')
    ax.set_title(title)

    ax.legend()

    if show:
        plt.show()


def latent_3d(x, y, ax=None, size=(12, 10), show=True):
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(projection='3d')

    for i, label in enumerate(reversed(np.unique(y))):
        x_ = x[y == label]
        ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2], s=10, label=utils.get_name(label))

    ax.set_xlabel(r'$z_0$')
    ax.set_ylabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    # ax.set_title(title)

    ax.legend()

    if show:
        plt.show()


def latent_kde(x: np.ndarray, y: np.ndarray, ax=None, size=(10, 10), limit=50_000):
    from scipy.stats import gaussian_kde

    def kde_hist(x, y, ax, ax_histx, ax_histy, **kwargs):
        x_ = np.sort(x)
        y_ = np.sort(y)

        # no labels
        ax_histx.tick_params(axis='x', labelbottom=False)
        ax_histy.tick_params(axis='y', labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, **kwargs)

        # kde histograms
        kde_x = gaussian_kde(x_)
        kde_y = gaussian_kde(y_)

        ax_histx.plot(x_, kde_x(x_))
        ax_histy.plot(kde_y(y_), y_)

    if ax is None:
        fig = plt.figure(figsize=size)
    else:
        fig = ax.figure

    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    if len(x) > limit:
        idx = np.random.choice(len(x), size=limit, replace=False)
        x = x[idx]
        y = y[idx]

    for i, label in enumerate(np.unique(y)):
        x_ = x[y == label]

        kde_hist(x_[:, 0], x_[:, 1], ax, ax_histx, ax_histy, s=10, alpha=0.7,
                 label=utils.get_name(label), zorder=-i)

    ax.set_xlabel(r'$z_0$')
    ax.set_ylabel(r'$z_1$')

    ax.legend()
    plt.show()


def latent_kde3d(x: np.ndarray, y: np.ndarray, ax=None, size=(10, 10), limit=50_000):
    from scipy.stats import gaussian_kde

    def kde_hist(x, y, z, ax, **kwargs):
        x_ = np.sort(x)
        y_ = np.sort(y)
        z_ = np.sort(z)

        # the scatter plot
        ax.scatter(x, y, z, **kwargs)

        # kde histograms
        kde_x = gaussian_kde(x_)
        kde_y = gaussian_kde(y_)
        kde_z = gaussian_kde(z_)

        p = ax.plot(x_, kde_x(x_), zs=min(ax.get_zlim()), zdir='z', linestyle='dotted')
        line_color = p[-1].get_color()

        ax.plot(y_, kde_y(y_), zs=min(ax.get_xlim()), zdir='x', linestyle='dotted',
                color=line_color)

        ax.plot(z_, kde_z(z_), zs=max(ax.get_ylim()), zdir='y', linestyle='dotted',
                color=line_color)

    if ax is None:
        fig = plt.figure(figsize=size)
    else:
        fig = ax.figure

    ax = fig.add_subplot(projection='3d')

    if len(x) > limit:
        idx = np.random.choice(len(x), size=limit, replace=False)

        x = x[idx]
        y = y[idx]

    for i, label in enumerate(np.unique(y)):
        x_ = x[y == label]

        kde_hist(x_[:, 0], x_[:, 1], x_[:, 2], ax, s=10, alpha=0.7,
                 label=utils.get_name(label), zorder=-i)

    ax.set_xlabel(r'$z_0$')
    ax.set_ylabel(r'$z_1$')
    ax.set_zlabel(r'$z_2$')

    ax.legend()
    plt.show()


# -------------------------------------------------------------------------------------------------
# -- Profile plots
# -------------------------------------------------------------------------------------------------

def compute_profile(x: np.ndarray, y: np.ndarray, num_bins=(100, 100)):
    # Source: https://vmascagn.web.cern.ch/LABO_2020/profile_plot.html
    # use of the 2d hist by numpy to avoid plotting
    h, x_edges, _ = np.histogram2d(x, y, bins=num_bins)

    # bin width
    bin_width = x_edges[1] - x_edges[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    # also the x values should be recomputed because of the possibility of empty slices
    x_array = []
    x_slice_mean = []
    x_slice_rms = []

    for i in range(x_edges.size - 1):
        y_value = y[(x > x_edges[i]) & (x <= x_edges[i + 1])]

        if y_value.size > 0: # do not fill the quantities for empty slices
            x_array.append(x_edges[i] + bin_width / 2)
            x_slice_mean.append(y_value.mean())
            x_slice_rms.append(y_value.std())

    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)

    return x_array, x_slice_mean, x_slice_rms


def plot_profile(x, y, ax, cls_label: str, x_label: str, y_label: str, num_bins=(100, 100),
                 bar_alpha=0.25, bar_color: str = None, bar_style: str = None, **kwargs):
    # Source: https://vmascagn.web.cern.ch/LABO_2020/profile_plot.html
    p_x, p_mean, p_rms = compute_profile(x, y, num_bins=num_bins)

    bars = ax.errorbar(p_x, p_mean, p_rms, fmt='_', label=cls_label, **kwargs)

    if isinstance(bar_color, str):
        bars[0].set_color(bar_color)

    if isinstance(bar_style, str):
        bars[-1][0].set_linestyle(bar_style)

    bars[-1][0].set_alpha(float(bar_alpha))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def profiles(data: dict, y: np.ndarray, m: np.ndarray, **kwargs):
    """Arranges profile plots comparing two variables `z` vs `x` (being keys of `data`),
       for each signal in `y` and across all mass points in `m`.
    """
    from ad.constants import BKG_INDEX
    classes = np.unique(y)

    z_name, z = data.get('z')
    x_name, x = data.get('x')

    axes = utils.get_plot_axes(rows=z.shape[-1], cols=len(classes) - 1)
    axes = np.reshape(axes, newshape=(z.shape[-1], -1))

    bkg_mask = np.squeeze(y == BKG_INDEX)

    for i in range(z.shape[-1]):
        zi = z[:, i]
        k = 0

        for label in classes:
            name = utils.get_name(label)

            if name == utils.get_bkg_name():
                continue

            # plot profiles for each signal mass
            ax = axes[i][k]

            plot_profile(x=x[bkg_mask], y=zi[bkg_mask], cls_label=utils.get_bkg_name(),
                         ax=ax, x_label=x_name, y_label=z_name, **kwargs)

            for mass, mass_name in utils.get_masses(label).items():
                mask = np.squeeze(m == mass)

                plot_profile(x=x[mask], y=zi[mask], ax=ax, cls_label=f'{name} ({mass_name})',
                             x_label=x_name, y_label=z_name, **kwargs)

            ax.legend(loc='upper right')
            k += 1

    plt.tight_layout()
    plt.show()

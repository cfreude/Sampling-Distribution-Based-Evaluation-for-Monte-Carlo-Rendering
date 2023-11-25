import numpy as np
import pickle, json
import scipy.stats
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
 
from statistics import Statistic, Samples, AD, ADdS, ADmS, RMSE, RelRMSE, SDMP, SDMP2, JHD20, Wasserstein
import compute
import common

def remove_ticks(ax):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


class OnlineViewer:

    def __init__(self, _column_tags, _export_path, _fig_save_interval=32):

        self.column_tags = _column_tags
        self.export_path = _export_path

        self.cols = len(self.column_tags)
        self.rows = -1

        self.distr_stats_dict = OrderedDict()
        self.ref_key = [None]
        self.ref_w = 0
        self.ref_h = 0

        self.fig = None
        self.axs = None
        self.hs = None
        self.initialized = False
        self.axes_set = False
        self.fig_save_interval = _fig_save_interval
        self.fig_save_counter = 1

        self.fig_label_size = 8
        self.label_pad = 4

        self.data_tranform_dict = {
            Statistic.MEAN_TAG: (lambda x: common.tone_map(x), lambda arr: 1.0),
            Statistic.STD_TAG: (lambda x: common.clip_percentile(x), lambda arr: np.max(arr)),
            Statistic.COFV_TAG: (lambda x: common.clip_percentile(x), lambda arr: np.max(arr)),
            AD.TAG: (lambda x: common.clip_percentile(x), lambda arr: np.max(arr)),
            ADdS.TAG: (lambda x: common.clip_percentile(x), lambda arr: np.max(arr)),
            ADmS.TAG: (lambda x: common.clip_percentile(x), lambda arr: np.max(arr)),
            RelRMSE.TAG: (lambda x: common.clip_percentile(x), lambda arr: np.nanmax(arr)),
            RMSE.TAG: (lambda x: common.clip_percentile(x), lambda arr: np.nanmax(arr)),
            SDMP.TAG: (lambda x: x, lambda arr: np.max(arr)),
            SDMP2.TAG: (lambda x: x, lambda arr: np.max(arr)),
            JHD20.TAG: (lambda x: x, lambda arr: np.max(arr)),
            Wasserstein.TAG: (lambda x: common.clip_percentile(x), lambda arr: np.max(arr)),
        }

    def init_figure(self):

        # TODO think about using img-grid
        self.fig, self.axs = plt.subplots(self.rows, self.cols, sharex=True, sharey=True, figsize=(10, 6), dpi=200)

        plt.subplots_adjust(left=0.03, bottom=0.01, right=0.999, top=0.95, wspace=0.01, hspace=0.3)
        params = {
            'axes.labelsize': self.fig_label_size,
            'axes.labelpad': self.label_pad,
            'axes.titlesize': self.fig_label_size,
            'axes.titlepad': self.label_pad+1,
            'figure.titlesize': self.fig_label_size,
            #'font.size': self.fig_label_size
        }
        plt.rcParams.update(params)

        self.axs = self.axs.flatten()

        self.hs = []

        rnd = np.repeat(np.atleast_3d(1.0-np.identity(32)*0.5), 3, axis=2)
        for ax in self.axs:
            #ax.axis('off')
            h = ax.imshow(np.zeros((1, 1, 3)), vmin=0.0, vmax=1.0, interpolation='none', aspect='equal')
            self.hs.append(h)
            h.set_data(rnd)
            remove_ticks(ax)

        self.attach_click()

        plt.ion()
        plt.show()

    def plot_func(self, axi, raw_data, task_key, value_key, norm_dict, _spp, _seed, _iter):
        data = np.copy(raw_data)
        avg = np.mean(data)
        if value_key in list(self.data_tranform_dict.keys()):
            trans_data = self.data_tranform_dict[value_key][0](data)
        mval = norm_dict[value_key]
        if mval > 0.0:
            trans_data /= mval
        self.hs[axi].set_data(trans_data)
        self.axs[axi].set_title('%s\n (avg. %.4f)' % (value_key, avg))

        if self.fig_save_counter >= self.fig_save_interval:
            if _iter is None:
                path = os.path.join(self.export_path, 'img-%s-%s' % (task_key, value_key))
            else:
                path = os.path.join(self.export_path, 'img-%s-%s-%d' % (task_key, value_key, _iter+1))

            plt.imsave(path+'.png', trans_data)
            np.save(path+'.npy', data)

            key = '%s-%s' % (task_key, value_key)
            data = (_iter if _iter is None else _iter+1, avg)
            dat_file = os.path.join(self.export_path, 'avg-vals.dat')
            if not os.path.exists(dat_file):
                common.save_obj({key:[data]}, dat_file)
            else:
                data_dict = common.load_obj(dat_file)
                if key in list(data_dict.keys()):
                    data_dict[key].append(data)
                else:
                    data_dict[key] = [data]
                common.save_obj(data_dict, dat_file)

    def iteration_callback(self, _i, _count, scene_dict, tasks, ref_data):

        if not self.initialized:
            self.rows = len(list(scene_dict.keys()))
            if ref_data is not None:
                self.rows += 1
            self.init_figure()
            self.initialized = True

        if ref_data is not None:
            ref_key, ref_n, ref_stats, ref_spp, ref_seed = ref_data
            self.ref_key[0] = ref_key
            self.distr_stats_dict[ref_key] = ref_stats.get_values()
            self.distr_stats_dict[ref_key]['N'] = ref_n

        for task_key, (scene, inst_dict, _, _) in list(tasks.items()):
            stat_inst = inst_dict[Samples]
            self.distr_stats_dict[task_key] = stat_inst.get_values()
            self.distr_stats_dict[task_key]['N'] = stat_inst.get_num_samples()

        stats_arr_dict = OrderedDict()
        # extract data
        for task_key, (scene, inst_dict, _, _) in list(tasks.items()):
            for stat_type, stat_inst in list(inst_dict.items()):
                for value_key, data in list(stat_inst.get_values().items()):
                    if value_key in self.column_tags:
                        # for normalization
                        if value_key not in stats_arr_dict:
                            stats_arr_dict[value_key] = []
                        stats_arr_dict[value_key].append(np.copy(data))

        normalization_dict = OrderedDict()
        for value_key, data_array in list(stats_arr_dict.items()):
            if value_key in self.column_tags:
                data_func, norm_func = self.data_tranform_dict[value_key]
                transf_data_array = np.array([data_func(x) for x in data_array])
                norm_fac = norm_func(transf_data_array)
                normalization_dict[value_key] = norm_fac

        # PLOT
        axi = 0
        if ref_data is not None:
            task_key, ref_n, ref_stats, ref_spp, ref_seed = ref_data
            task_key += ' (N=%d)' % ref_n
            for value_key, raw_data in list(ref_stats.get_values().items()):
                if value_key in self.column_tags:
                    self.plot_func(axi, raw_data, task_key, value_key, normalization_dict, ref_spp, ref_seed, ref_n if _i == 0 else None)
                    axi += 1

            ref_tags = list(ref_stats.get_values().keys())
            off = [tag in self.column_tags for tag in ref_tags]
            off = np.sum(off)
            axi += self.cols - off

        for task_key, (scene, inst_dict, spp, seed) in list(tasks.items()):
            for stat_type, stat_inst in list(inst_dict.items()):
                for value_key, raw_data in list(stat_inst.get_values().items()):
                    if value_key in self.column_tags:
                        self.plot_func(axi, raw_data, task_key, value_key, normalization_dict, spp, seed, _i)
                        axi += 1

        axi = 0
        self.axs[axi].set_ylabel('%s (N: %d)\nspp: %d, seed: %d' % (ref_key, ref_n, ref_spp, ref_seed), size=self.fig_label_size, labelpad=self.label_pad)
        for task_key, (scene, inst_dict, spp, seed) in list(tasks.items()):
            axi += self.cols
            self.axs[axi].set_ylabel('%s\nspp: %d, seed: %d' % (task_key, spp, seed), size=self.fig_label_size, labelpad=self.label_pad)

        self.fig.suptitle('Samples: %d/%d' % (_i + 1, _count))

        if self.fig_save_counter >= self.fig_save_interval:
            path = os.path.join(self.export_path, 'plot-%d.pdf' % (_i + 1))
            plt.savefig(path, dpi=100)
            self.fig_save_counter = 0

        self.fig_save_counter += 1

        plt.pause(.05)


    @staticmethod
    def plot_b_in_a(ax, x, label, color, mean, std, n, alpha, b_stats=None):
        halpha = alpha * 0.5
        smp_distr_a = scipy.stats.norm(mean, std / np.sqrt(n))
        min_a, max_a = smp_distr_a.ppf(halpha), smp_distr_a.ppf(1.0 - halpha)
        x_a = np.linspace(min_a, max_a, len(x))
        if b_stats is None:
            distr_a = scipy.stats.norm(mean, std)
            ax.plot(x, distr_a.pdf(x), ':', color=color, label=label)
            ax.plot(x, smp_distr_a.pdf(x), color=color, label='%s, n=%d' % (label, n))
            ax.fill_between(x_a, 0, smp_distr_a.pdf(x_a), color=color, alpha=0.5)
        else:
            mean_b, std_b, n_b = b_stats
            distr_b = scipy.stats.norm(mean_b, std_b)
            smp_distr_b = scipy.stats.norm(mean_b, std_b / np.sqrt(n_b))
            prob_b_in_a = smp_distr_b.cdf(max_a) - smp_distr_b.cdf(min_a)
            ax.plot(x, distr_b.pdf(x), ':', color=color, label=label)
            ax.plot(x, smp_distr_b.pdf(x), color=color, label='%s, n=%d, P=%.4f' % (label, n_b, prob_b_in_a))
            ax.fill_between(x_a, 0, smp_distr_b.pdf(x_a), color=color, alpha=0.5)

    def attach_click(self):

        ref_key_list = self.ref_key
        stats = self.distr_stats_dict

        def onclick(event):
            ref_key = ref_key_list[0]
            ref_mean = stats[ref_key][Statistic.MEAN_TAG]

            # compute pixel coords
            w, h, _ = ref_mean.shape
            # need to swap
            y = int(np.floor((event.xdata + 0.5) * w))
            x = int(np.floor((event.ydata + 0.5) * h))

            ref_mean = ref_mean[x, y, :]
            ref_std = stats[ref_key][Statistic.STD_TAG][x, y, :]
            ref_n = stats[ref_key]['N']

            # compute maximum values for plotting
            min_vals = [np.inf, np.inf, np.inf]
            max_vals = [0, 0, 0]
            for k, v in list(stats.items()):
                mean = v[Statistic.MEAN_TAG][x, y, :]
                std = v[Statistic.STD_TAG][x, y, :]
                n = v['N']
                for i, c in enumerate(('red', 'green', 'blue')):
                    p001 = scipy.stats.norm(mean[i], std[i] / np.sqrt(n)).ppf(0.001)
                    if p001 < min_vals[i]:
                        min_vals[i] = p001
                    p999 = scipy.stats.norm(mean[i], std[i] / np.sqrt(n)).ppf(0.999)
                    if p999 > max_vals[i]:
                        max_vals[i] = p999

            # plot
            xv = [np.linspace(miv, mav, 200) for (miv, mav) in zip(min_vals, max_vals)]
            fig, axs = plt.subplots(1, 3, figsize=(17, 5))
            axs = axs.flatten()
            titles = ['RED\n', 'GREEN\n', 'BLUE\n', ]
            for ci, (task_key, v) in enumerate(stats.items()):
                mean = v[Statistic.MEAN_TAG][x, y, :]
                std = v[Statistic.STD_TAG][x, y, :]
                n = v['N']
                for i, c in enumerate(titles):
                    cxv = xv[i]
                    color = 'C%d' % ci
                    if task_key != ref_key:
                        OnlineViewer.plot_b_in_a(axs[i], cxv, task_key, color, ref_mean[i], ref_std[i], ref_n, 0.05, b_stats=(mean[i], std[i], n))
                    else:
                        OnlineViewer.plot_b_in_a(axs[i], cxv, task_key, color, ref_mean[i], ref_std[i], ref_n, 0.05, b_stats=None)
                    #    titles[i] += task_key + ' # '
                    #    titles[i] += ' | '.join(['%s: %.4f' % (tag, v[tag][x, y, i]) for tag in ['MSE', 'RelMSE', 'BiasProb']])
                    #    titles[i] += '\n'

            for i, title in enumerate(titles):
                axs[i].set_title(title)
                axs[i].legend()
            fig.suptitle('pixel coord: %s' % str((x, y)))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=None, hspace=0.4)
            #plt.rcParams.update({'font.size': 8})
            plt.show(block=False)

        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)


class SampleViewer:

    def __init__(self):
        self.sample_dict = OrderedDict()
        self.hist = True
        self.pixel = (64, 64)

        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 8))
        self.axs = self.axs.flatten()
        for ax in self.axs:
            ax.hist(np.random.rand(100), bins=10)
        plt.ion()
        plt.show()

    def sample_callback(self, _i, _count, _name, _sample):
        if _name not in list(self.sample_dict.keys()):
            self.sample_dict[_name] = []
        self.sample_dict[_name].append(_sample[self.pixel + (slice(None, None, None), )])

        self.fig.suptitle('Sample iteration: %d/%d' % (_i + 1, _count))

        for ax in self.axs:
            ax.cla()

        for ic, color in enumerate(['RED', 'GREEN', 'BLUE']):
            self.axs[ic].set_title('Hist. - '+color)
            self.axs[ic+3].set_title('eCDF - '+color)

        for i, (name, samples) in enumerate(self.sample_dict.items()):
            if len(samples) > 3:
                axi = 0
                for ic, color in enumerate(['RED', 'GREEN', 'BLUE']):
                    vals = np.array(samples)[:, ic]
                    m = np.mean(vals)
                    s = np.std(vals, ddof=1)
                    d = scipy.stats.norm(m, s)
                    # HIST
                    x = np.linspace(d.ppf(0.01) - s * 0.2, d.ppf(0.99) + s * 0.2, 200)
                    self.axs[axi].hist(vals, bins=int(np.sqrt(vals.shape[0])), label=name, alpha=0.5, normed=True, color='C%d' % (i+1))
                    self.axs[axi].plot(x, d.pdf(x), label=name, color='C%d' % (i+1))
                    # ECDF
                    x = np.sort(vals)
                    y = np.arange(vals.shape[0])/float(vals.shape[0])
                    self.axs[axi+3].plot(x, d.cdf(x), '--', label=name, color='C%d' % (i+1))
                    self.axs[axi+3].plot(x, y, '-', label=name, color='C%d' % (i+1))
                    axi += 1

        for ax in self.axs:
            ax.legend()


if __name__ == '__main__':

    export_path = './visualize-export'
    export_interval = 512

    if not os.path.exists(export_path):
        os.mkdir(export_path)

    reference_parameters = (1024, 0)
    num_iterations = 1024
    distance_measures = [AD, RMSE, SDMP, JHD20]

    scene_paths = [
        './data/veach-ajar/scene-ref.xml',
        './data/veach-ajar/scene-control.xml',
        './data/veach-ajar/scene-biased.xml',
        ]

    olv = OnlineViewer([Statistic.MEAN_TAG, Statistic.STD_TAG] + [d.TAG for d in distance_measures], _export_path=export_path, _fig_save_interval=export_interval)

    spf = None#SampleViewer().sample_callback
    compute.compute_test(reference_parameters, num_iterations, scene_paths, distance_measures, _spp=32, _seed=1, _iteration_callback=olv.iteration_callback, _sample_callback=spf)
    plt.show(block=True)
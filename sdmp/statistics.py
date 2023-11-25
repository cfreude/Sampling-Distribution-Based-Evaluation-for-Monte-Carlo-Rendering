import numpy as np
from collections import OrderedDict
import scipy.stats as st
from scipy.special import erf, erfc, erfcinv
from skimage.metrics import structural_similarity

from common import OnlineStats


class Statistic(object):

    mean = None
    std = None
    cofv = None
    n = None

    MEAN_TAG = 'Mean'
    STD_TAG = 'STD'
    COFV_TAG = 'Cofv'

    def __init__(self):
        pass

    def push(self, _sample):
        pass

    def set(self, _ref_mean, _ref_std, _ref_n):
        self.mean = _ref_mean
        self.std = _ref_std
        self.n = _ref_n
        self.cofv = np.copy(self.std)
        np.divide(self.std, self.mean, out=self.cofv, where=self.mean > 0)

    def get_values(self):
        return OrderedDict([
            (Statistic.MEAN_TAG, self.mean),
            (Statistic.STD_TAG, self.std),
            (Statistic.COFV_TAG, self.cofv)
        ])

    def get_num_samples(self):
        return self.n


class Samples(Statistic):

    def __init__(self):
        super(Statistic, self).__init__()
        self.stats = OnlineStats()

    def push(self, _sample):
        self.stats.push(_sample)

    def get_values(self):
        cofv = np.copy(self.stats.standard_deviation())
        np.divide(self.stats.standard_deviation(), self.stats.mean(), out=cofv, where=self.stats.mean() > 0)
        return OrderedDict([
            (Statistic.MEAN_TAG, self.stats.mean()),
            (Statistic.STD_TAG, self.stats.standard_deviation()),
            (Statistic.COFV_TAG, cofv)
            ])

    def get_num_samples(self):
        return self.stats.num_data_values()


class AD(Samples):

    TAG = 'AD'

    def __init__(self):
        super(AD, self).__init__()

    def get_values(self):
        if self.stats.num_data_values() > 0:
            d = np.abs(self.stats.mean() - self.mean)
        else:
            d = None
        return OrderedDict([(AD.TAG, d)])


class ADdS(Samples):

    TAG = 'ADdS'

    def __init__(self):
        super(ADdS, self).__init__()

    def get_values(self):
        if self.stats.num_data_values() > 0:
            d = np.abs(self.stats.mean() - self.mean)
            nom = self.stats.standard_deviation() + self.std
            nom = np.divide(self.stats.standard_deviation() * self.std, nom, where=nom > 0)
            d = np.divide(d, nom, where=nom > 0)
        else:
            d = None
        return OrderedDict([(ADdS.TAG, d)])


class ADmS(Samples):

    TAG = 'ADmS'

    def __init__(self):
        super(ADmS, self).__init__()

    def get_values(self):
        if self.stats.num_data_values() > 0:
            d = np.abs(self.stats.mean() - self.mean) - self.stats.standard_deviation() * self.std
        else:
            d = None
        return OrderedDict([(ADmS.TAG, d)])


class RMSE(Samples):

    TAG = 'RMSE'

    def __init__(self):
        super(RMSE, self).__init__()

    def push(self, _sample):
        d = np.square(_sample - self.mean)
        self.stats.push(d)

    def get_values(self):
        return OrderedDict([(RMSE.TAG, np.sqrt(self.stats.mean()))])


class RelRMSE(Samples):
    # https://www.cs.umd.edu/~zwicker/publications/AdaptiveSamplingGreedyError-SIGA11.pdf

    TAG = 'RelRMSE'

    def __init__(self):
        super(RelRMSE, self).__init__()

    def push(self, _sample):
        d = np.square(_sample - self.mean)
        self.stats.push(d)

    def get_values(self):
        if self.mean is None:
            denom = 1.0
        else:
            eps = np.finfo(self.stats.dtype).eps
            denom = self.mean + eps
        return OrderedDict([(RelRMSE.TAG, np.sqrt(self.stats.mean()) / denom)])


class SDMP(Samples):

    TAG = 'SDMP'

    def __init__(self):
        super(SDMP, self).__init__()

    def push(self, _sample):
        self.stats.push(_sample)

    @staticmethod
    def _func(_func, m0, s0, m1, s1, x):
        sq2 = np.sqrt(2)
        return _func((m1 - m0 + sq2 * s0 * erfcinv(x)) / (sq2 * s1))

    @staticmethod
    def func(m0, s0, m1, s1, a):
        #sq2 = np.square(2)
        #fac = lambda m0, s0, m1, s1, x: erfc((m0 - m1 + sq2 * s1 * erfcinv(x)) / (sq2 * s0))
        #out = 0.5 * (fac(m0, s0, m1, s1, 2.0 - 2.0 * a) - fac(m0, s0, m1, s1, 2.0 * a))

        out = 0.5 * (1 + SDMP._func(erf, m0, s0, m1, s1, 2.0 - a) + SDMP._func(erfc, m0, s0, m1, s1, a))

        dz = (m0-m1) == 0.0
        s0z = s0 == 0.0
        s1z = s1 == 0.0
        bsz = np.logical_and(s0z, s1z)
        all_zero = np.logical_and(dz, bsz)
        out = np.atleast_1d(out)
        out[all_zero] = a
        return out

    def get_values(self):

        if self.get_num_samples() > 1:
            alpha = 0.05
            out = SDMP.func(
                self.mean,
                self.std / np.sqrt(self.n),
                self.stats.mean(),
                self.stats.standard_deviation() / np.sqrt(self.stats.num_data_values()),
                alpha)

            is_nan = np.isnan(out)
            if np.any(is_nan):
                out[is_nan] = 0.0
        else:
            out = np.zeros((1, 1, 3))

        return OrderedDict([(SDMP.TAG, out)])


class SDMP2(Samples):

    TAG = 'SDMP2'

    def __init__(self):
        super(SDMP2, self).__init__()

    def push(self, _sample):
        self.stats.push(_sample)

    def get_values(self):

        if self.get_num_samples() > 1:
            alpha = 0.05

            ref_sdm_std =  self.std / np.sqrt(self.n)
            test_sdm_std = self.stats.standard_deviation() / np.sqrt(self.stats.num_data_values())

            out_a = SDMP.func(
                self.mean,
                ref_sdm_std,
                self.stats.mean(),
                test_sdm_std,
                alpha)

            is_nan = np.isnan(out_a)
            if np.any(is_nan):
                out_a[is_nan] = 0.0

            out_b = SDMP.func(
                self.stats.mean(),
                test_sdm_std,
                self.mean,
                ref_sdm_std,
                alpha)

            is_nan = np.isnan(out_b)
            if np.any(is_nan):
                out_b[is_nan] = 0.0

            out = (out_a + out_b) * 0.5
        else:
            out = np.zeros((1, 1, 3))

        return OrderedDict([(SDMP2.TAG, out)])


class JHD20(Samples):

    TAG = 'JHD20'

    def __init__(self):
        super(JHD20, self).__init__()

    def push(self, _sample):
        self.stats.push(_sample)

    @staticmethod
    def func(mx, vx, my, vy, nx, ny, tails=2):
        """
        Welch's t-test for two unequal-size samples, not assuming equal variances
        """
        assert tails in (1, 2), "invalid: tails must be 1 or 2, found %s" % str(tails)
        df = ((vx / nx + vy / ny) ** 2 /  # Welch-Satterthwaite equation
              ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)))
        t_obs = (mx - my) / np.sqrt(vx / nx + vy / ny)
        p_value = tails * st.t.sf(abs(t_obs), df)
        return p_value

    def get_values(self):

        if self.get_num_samples() > 1:
            out = JHD20.func(
                self.mean,
                self.std**2,
                self.stats.mean(),
                self.stats.standard_deviation()**2,
                self.n,
                self.stats.num_data_values())

            is_nan = np.isnan(out)
            if np.any(is_nan):
                out[is_nan] = 0.0
        else:
            out = np.zeros((1, 1, 3))

        return OrderedDict([(JHD20.TAG, out)])


class Wasserstein(Samples):

    TAG = 'WaSt'

    def __init__(self):
        super(Wasserstein, self).__init__()

    def push(self, _sample):
        self.stats.push(_sample)

    @staticmethod
    def func(m0, s0, m1, s1, d=2.0):
        return np.power(np.square(m0 - m1) + np.square(s0 - s1), 1.0 / d)# - np.power(np.square(s0-s1), 1.0/d)

    def get_values(self):

        if self.get_num_samples() > 1:
            out = Wasserstein.func(self.mean, self.std, self.stats.mean(), self.stats.standard_deviation())
        else:
            out = np.ones((1, 1, 3))

        return OrderedDict([(Wasserstein.TAG, out)])


class SSIM(Samples):

    TAG = 'SSIM'

    def __init__(self):
        super(SSIM, self).__init__()

    def get_values(self):
        if self.stats.num_data_values() > 0:
            _, _, d = structural_similarity(self.mean, self.stats.mean(), full=True)
        else:
            d = None
        return OrderedDict([(SSIM.TAG, d)])


if __name__ == '__main__':

    import csv
    import matplotlib.pyplot as plt

    from scipy.stats import norm, beta
    from scipy.stats import ttest_ind

    a = beta(0.5, 1)
    b = beta(0.7, 1)

    x = np.linspace(0, 1, 100)
    plt.plot(x, a.pdf(x))
    plt.plot(x, b.pdf(x))
    plt.show()

    samples = [(128, 128), (1024*4, 128), (1024*4, 1024*4)]
    for (na, nb) in samples:

        sa = a.rvs(na)
        sb = b.rvs(nb)

        t, p = ttest_ind(sa, sb, equal_var=False, axis=None); print(p)
        wp = JHD20.func(np.mean(sa), np.var(sa, ddof=1), np.mean(sb), np.var(sb, ddof=1), na, nb, 2); print(wp)
        sp = SDMP.func(np.mean(sa), np.std(sa, ddof=1) / np.sqrt(na), np.mean(sb), np.std(sb, ddof=1) / np.sqrt(nb), 0.05); print(sp)

        with open('./biased-%d-%d-samples.csv' % (na, nb), mode='wb') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([na, ] + list(sa.tolist()))
            writer.writerow([nb, ] + list(sb.tolist()))
            writer.writerow([sp[0], wp])

import numpy as np
import scipy.stats


class OnlineMoments(object):

    def __init__(self, dtype=np.float):
        self.n = 0
        self.m = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0
        self.dtype = dtype

    def push(self, sample):
        # convert to be safe
        sample = np.array(sample, dtype=self.dtype)

        self.n += 1

        d = sample - self.m
        d2 = d * d
        dN = d / self.n
        dN2 = dN * dN

        self.m = self.m + dN
        self.m2 = self.m2 + d * (d - dN)
        self.m3 = self.m3 - 3. * dN * self.m2 + d * (d2 - dN2)
        self.m4 = self.m4 - 4. * dN * self.m3 - 6. * dN2 * self.m2 + d * (d * d2 - dN * dN2)

    def mean(self):
        return self.m

    def variance(self):
        return self.m2 / (self.n - 1)

    def skewness(self):
        return self.m3 / (self.n * np.power(self.m2 / self.n, 3. / 2.))

    def kurtosis(self):
        return self.m4 * self.n / (self.m2 * self.m2)

    def moments(self):
        moments = np.array([np.zeros(self.m.shape), self.m2, self.m3, self.m4])
        return moments / self.n

    def moment(self, i=1):
        return self.moments()[i-1]


if __name__ == '__main__':

    np.random.seed(0)
    vals = np.random.rand(10, 2)

    vals = [0.1354770043, 0.83500859, 0.9688677711, 0.221034043, 0.3081670505, 0.5472205964, 0.188381976, 0.9928813019,
            0.9964613255, 0.967694937, 0.7258389632, 0.9811096918, 0.1098617508, 0.7981058567, 0.2970294496,
            0.004783484419, 0.1124645161, 0.6397633571, 0.8784306454, 0.5036626777, 0.7979286152, 0.3612940013,
            0.2119243324, 0.6813595386, 0.3987385199, 0.7406472447, 0.4747586806, 0.4220876811, 0.173865172,
            0.3019131269, 0.7972799152, 0.3165504448, 0.8724288201, 0.1491139764, 0.9940684943, 0.8219032648,
            0.1251827645, 0.7637500126, 0.4905890396, 0.6636055205, 0.1258966335, 0.2102090745, 0.05121642579,
            0.03644125159, 0.408731161, 0.4579891554, 0.4875689269, 0.7939749715, 0.9208747912, 0.8075310254,
            0.7057742517, 0.002818432562, 0.7107038751, 0.6439609565, 0.4560328245, 0.7739171289, 0.5737546666,
            0.8767574151, 0.8081754901, 0.01777389558, 0.8212459916, 0.8208407842, 0.9400740288, 0.4126665149,
            0.4231651164, 0.5809566777, 0.1580575846, 0.7617312137, 0.2301560645, 0.8097345487, 0.9885216008,
            0.3324482823, 0.2998317058, 0.01353912667, 0.2172378395, 0.9073647178, 0.848467792, 0.9550175735,
            0.7788977101, 0.9874596269, 0.06759538114, 0.7935975815, 0.5945035612, 0.7327987253, 0.6952328838,
            0.6798197907, 0.3923204692, 0.5615574424, 0.2080680571, 0.5273714586, 0.4042085181, 0.3527624081,
            0.5928238785, 0.3563451606, 0.9649663721, 0.1544384174, 0.3949082106, 0.3872959051, 0.7269547216,
            0.3885698075]

    om = OnlineMoments()
    for v in vals:
        om.push(v)

    print('mean:', np.mean(vals, axis=0), om.mean())
    print('variance:', np.var(vals, ddof=1, axis=0), om.variance())
    print('skewness:', scipy.stats.skew(vals, axis=0), om.skewness())
    print('kurtosis:', scipy.stats.kurtosis(vals, fisher=False), om.kurtosis())
    for i in range(1, 5):
        print('%d moment:' % i, scipy.stats.moment(vals, i), om.moment(i))




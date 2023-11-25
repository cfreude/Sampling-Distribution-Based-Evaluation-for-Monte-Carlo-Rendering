import os
import numpy as np
from collections import OrderedDict
from statistics import Statistic, Samples
import common

import mitsuba as mi
mi.set_variant('scalar_rgb')


def print_results(_i, _tasks, _distance_measures):
    col_width = 15
    col_names = [list(d().get_values().keys())[0] for d in _distance_measures]
    row_format = ('{:<%d} ' % (col_width-1)) * (len(col_names) + 1)
    print('#' * col_width * (len(col_names) + 1))
    print(row_format.format('ITERATION #%s' % _i, *col_names))
    for scene_name, (scene, stats, spp, seed) in list(_tasks.items()):
        avg_values = []
        for stat_type, inst in list(stats.items()):
            if stat_type != Samples:
                for value_name, value in list(inst.get_values().items()):
                    avg = np.nanmean(value)
                    avg_values.append(avg)
        print(row_format.format('%s (%d SPP, seed: %d)' % (scene_name, spp, seed), *avg_values))
    print('#' * col_width * (len(col_names) + 1))



@common.cache_to_disk('./cache/')
def compute_reference(_num_iterations, _ref_scene_path, _file_hash, _spp, _seed):

    print('Computing reference data ...')
    #mts = PyMts(CPU_COUNT)
    #scn = mts.load_scene_custom(_ref_scene_path, _spp=_spp, _seed=_seed)
    scn = mi.load_file(_ref_scene_path, parallel=True, spp=_spp)
    print(_spp)

    stats = common.OnlineStats()

    timer = common.Timer()
    timer.start()

    for i in range(_num_iterations):
        rend_a = mi.render(scn, spp=_spp, seed=_seed+i)
        #print(hash(rend_a))
        #mi.util.convert_to_bitmap(rend_a).write(f'./{i}.png')
        stats.push(rend_a)

        print(timer.eta(i+1, _num_iterations))

    timer.stop()

    return stats.mean(), stats.standard_deviation()


def compute_test(_reference_parameters, _num_iterations, _scene_paths, _distance_measures, _spp=32, _seed=0, _iteration_callback=None, _sample_callback=None):

    if _reference_parameters is None:
        assert len(_scene_paths) == 2, 'too many scenes for non-reference mode'
    else:
        # prepare reference
        ref_n, ref_ind = _reference_parameters
        ref_seed = _seed+1
        ref_path = _scene_paths[ref_ind]
        hash = common.hash_file(ref_path)
        ref_mean, ref_std = compute_reference(ref_n, ref_path, hash, _spp, ref_seed)
        ref_coov = np.copy(ref_std)
        np.divide(ref_std, ref_mean, out=ref_coov, where=ref_mean > 0.0)
        ref_stats = Statistic()
        ref_stats.set(ref_mean, ref_std, ref_n)
        scene_name = os.path.basename(os.path.splitext(ref_path)[0])
        ref_data = (scene_name, ref_n, ref_stats, _spp, ref_seed)

    # config
    scene_dict = OrderedDict()
    for i, path in enumerate(_scene_paths):
        if _reference_parameters is not None and i == ref_ind:
            continue
        scene_name = os.path.basename(os.path.splitext(path)[0])
        scene_dict[scene_name] = path

    sample_processors = [Samples] + _distance_measures

    # load scenes and create distance class instances
    tasks = OrderedDict()
    scene_count = len(list(scene_dict.keys()))
    seeds = np.random.choice(np.arange(ref_seed+1, ref_seed+1+scene_count*1000), size=(scene_count, ), replace=False)
    for i, (scene_name, path) in enumerate(scene_dict.items()):
        #seed = i+_seed+2
        seed = seeds[i]
        #scene = mts.load_scene_custom(path, _spp=_spp, _seed=seed)
        scene = mi.load_file(path, parallel=True, spp=_spp)
        stats_dict = OrderedDict()
        for classtype in sample_processors:
            instance = classtype()
            if _reference_parameters is not None:
                instance.set(ref_mean, ref_std, ref_n)
            stats_dict[classtype] = instance
        tasks[scene_name] = (scene, stats_dict, _spp, seed)

    # iterate
    timer = common.Timer()
    timer.start()
    for i in range(_num_iterations):

        # get samples
        sample_dict = OrderedDict()
        for scene_name, (scene, stats, _, s_seed) in list(tasks.items()):
            sample_dict[scene_name] = mi.render(scene, seed=s_seed+i)

        # push into Samples
        for scene_name, (scene, stats, _, _) in list(tasks.items()):
            stats[Samples].push(sample_dict[scene_name])
            if _sample_callback is not None:
                _sample_callback(i, _num_iterations, scene_name, sample_dict[scene_name])

        # use each Samples as reference for the other
        if _reference_parameters is None:
            scene_name_arr = list(tasks.keys())
            assert len(scene_name_arr) == 2, 'too many scenes for non-reference mode'
            ref_stats = tasks[scene_name_arr[0]][1][Samples]
            for stat_type, inst in list(tasks[scene_name_arr[1]][1].items()):
                if stat_type != Samples:
                    vals = ref_stats.get_values()
                    inst.set(vals[Samples.MEAN_TAG], vals[Samples.STD_TAG], ref_stats.get_num_samples())
            ref_stats = tasks[scene_name_arr[1]][1][Samples]
            for stat_type, inst in list(tasks[scene_name_arr[0]][1].items()):
                if stat_type != Samples:
                    vals = ref_stats.get_values()
                    inst.set(vals[Samples.MEAN_TAG], vals[Samples.STD_TAG], ref_stats.get_num_samples())
            ref_stats = None

        # process samples
        for scene_name, (scene, stats, _, _) in list(tasks.items()):
            for stat_type, inst in list(stats.items()):
                if stat_type != Samples:
                    inst.push(sample_dict[scene_name])

        if _iteration_callback is not None:
            _iteration_callback(i, _num_iterations, scene_dict, tasks, ref_data)

        print_results(i+1, tasks, _distance_measures)

        print(timer.eta(i+1, _num_iterations))

    timer.stop()

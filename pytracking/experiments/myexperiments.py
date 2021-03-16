from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset


def dimp50_lasot():
    trackers = trackerlist('dimp', 'super_dimp', range(1))

    dataset = get_dataset('lasot')
    return trackers, dataset


def trdimp_lasot():
    trackers = trackerlist('trdimp', 'trdimp_lasot', range(1))

    dataset = get_dataset('lasot')
    return trackers, dataset


def trsiam_lasot():
    trackers = trackerlist('trdimp', 'trsiam_lasot', range(1))

    dataset = get_dataset('lasot')
    return trackers, dataset


def trdimp_otb():
    trackers = trackerlist('trdimp', 'trdimp', range(1))

    dataset = get_dataset('otb')
    return trackers, dataset


def trsiam_otb():
    trackers = trackerlist('trdimp', 'trsiam', range(1))

    dataset = get_dataset('otb')
    return trackers, dataset


def trdimp_trackingnet():
    trackers = trackerlist('trdimp', 'trdimp', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset


def trsiam_trackingnet():
    trackers = trackerlist('trdimp', 'trsiam', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset
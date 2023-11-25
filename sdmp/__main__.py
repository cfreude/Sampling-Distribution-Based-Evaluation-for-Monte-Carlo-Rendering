import argparse
from compute import compute_test
from statistics import SDMP

parser = argparse.ArgumentParser(
    description='Compute the "SDMP" distance between MC renderings (via Mitsuba 3).'
)

parser.add_argument('spp',
                    metavar='spp',
                    type=int,
                    help='Defines the number of samples per pixel computed for each rendering.',
                    )

parser.add_argument('refc',
                    metavar='ref-num-iter',
                    type=int,
                    help='Defines the number of individual renderings computed for the reference.',
                    )

parser.add_argument('refp',
                    metavar='ref-scene-path',
                    type=str,
                    help='Defines the path to the Mitsuba XML scene file used to compute the reference data.'
                    )

parser.add_argument('testc',
                    metavar='test-num-iter',
                    type=int,
                    help='Defines the number of individual renderings computed for the test scenes.',
                    )

parser.add_argument('testps',
                    metavar='test-scene-paths',
                    type=str,
                    nargs='+',
                    help='Defines the paths to the Mitsuba XML scene file used to compute the test data.'
                    )

args = parser.parse_args()
print(args.testps)
compute_test((args.refc, 0), args.testc, [args.refp,]+args.testps, [SDMP], args.spp)
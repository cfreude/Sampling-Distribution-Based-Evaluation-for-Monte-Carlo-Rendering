# SDM-based probability (SDMP)

## Description

An implementation of the "Sampling Distribution of the Mean"-based probability (SDMP).
It enables the comparison of MC rendering algorithms (using Mitsuba) as outlined in the corresponding [paper](https://www.cg.tuwien.ac.at/research/publications/2023/freude-2023-sem/).

## Getting Started

### Dependencies

Python 3:
- numpy
- scipy
- matplotlib
- scikit-image
- scikit-learn
- mitsuba

### Installing

1. Setup your Python installation.
(for Anaconda users see .conda.yml)

2. Copy or install the this module.

### Executing program

1. Setup a Mistuba XML file as the reference, e.g. "./data/veach-ajar/scene-ref.xml".
2. Setup Mistuba XML files for the test / comparison, e.g. "./data/veach-ajar/scene-control.xml" and "./data/veach-ajar/scene-biased.xml"
3. Make sure sampler parameters match for all of the used scenes.
4. Choose SPP (>32 recommneded) and iteration count (e.g. 1024 for the reference and 32 for the test runs) and execute:

```
python -m sdmp 32 1024 ./data/veach-ajar/scene-ref.xml 256 ./data/veach-ajar/scene-control.xml ./data/veach-ajar/scene-biased.xml
```

For detailed command line parameters see help:
```
python -m sdmp -h
``` 

## Authors

Christian Freude, freude (at) cg.tuwien.ac.at

## Version History

* 0.9.0
    * Initial Release

## License

This project is licensed under the GNU GPL LICENSE - see the LICENSE.md file for details

## Acknowledgments

Funded by Austrian Science Fund (FWF): ORD 61
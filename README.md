# Huawei RAMP 2021 GOLDITE PESCE RHIM
Repository for the Huawei RAMP 2021 competition (OAN failure challenge).
Authors: Valentin GOLDITE Florian PESCE Ayoub RHIM
## Install dependencies
Run this line in the repository folder:
```
$ pip install -r requirements.txt
```
Main package installed are:
- [ramp-workflow ](https://github.com/ramp-kits/oan_failure) project
- [oan_failure](https://github.com/ramp-kits/oan_failure) project
- pytorch-cpu (1.9)
- lightgbm
- pandas
- scikit-learn
- matplotlib

## Download and prepare data
First download the file **public_data.tar.bz2** from the slack channel *#oan_failure_challenge*

Unzip content in **Huawei2021** folder.

Run prepare data script:
```
$ python data/prepare_data.py
```
Moove city_A and city_B folder in the data folder:
```
$ mv -r city_A data/city_A
$ mv -r city_B data/city_B
```
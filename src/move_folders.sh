#!/bin/bash

dataset_path="/home/msiau/data/tmp/jesmoris/spherical_coefficients_L50"
new_dataset_path="/home/msiau/data/tmp/jesmoris/spherical_coefficients_L50_dw"

rm -r -f $new_dataset_path
mkdir $new_dataset_path
mkdir $new_dataset_path"/Dundee"
mkdir $new_dataset_path"/Orkney"
mkdir $new_dataset_path"/Dundee/6ROW Bere"
mkdir $new_dataset_path"/Orkney/6ROW Bere"

# Dundee
cp -r $dataset_path"/Dundee 2ROW British" $new_dataset_path"/Dundee/2ROW British"
cp -r $dataset_path"/Dundee 2ROW Scottish" $new_dataset_path"/Dundee/2ROW Scottish"
cp -r $dataset_path"/Dundee 6ROW BERE Orkney" $new_dataset_path"/Dundee/6ROW Bere/6ROW BERE Orkney"
cp -r $dataset_path"/Dundee 6ROW BERE Unknown" $new_dataset_path"/Dundee/6ROW Bere/6ROW BERE Unknown"
cp -r $dataset_path"/Dundee 6ROW BERE Western Isles" $new_dataset_path"/Dundee/6ROW Bere/6ROW BERE Western Isles"
cp -r $dataset_path"/Dundee 6ROW Faro" $new_dataset_path"/Dundee/6ROW Faro"
cp -r $dataset_path"/Dundee 6ROW Scandinavian" $new_dataset_path"/Dundee/6ROW Scandinavian"
# Orkney
cp -r $dataset_path"/Orkney 2ROW British" $new_dataset_path"/Orkney/2ROW British"
cp -r $dataset_path"/Orkney 2ROW Scottish" $new_dataset_path"/Orkney/2ROW Scottish"
cp -r $dataset_path"/Orkney 6ROW BERE ORKNEY" $new_dataset_path"/Orkney/6ROW Bere/6ROW BERE ORKNEY"
cp -r $dataset_path"/Orkney 6ROW BERE Unknown" $new_dataset_path"/Orkney/6ROW Bere/6ROW BERE Unknown"
cp -r $dataset_path"/Orkney 6ROW BERE Western Isles" $new_dataset_path"/Orkney/6ROW Bere/6ROW BERE Western Isles"
cp -r $dataset_path"/Orkney 6ROW Scandinavian" $new_dataset_path"/Orkney/6ROW Scandinavian"

#!/bin/bash



echo Extracting models

cd ~;

mkdir pscdata;
cd pscdata;

scp -r guilherme.vieira-manhaes@vanda.polytechnique.fr:~/*.fr .;

mkdir finals results;
mmv "*.fr/psc/whatIs/finals/*.pkl" "./finals/#2_17042023_#1.pkl";
mmv "*.fr/psc/whatIs/results.csv" "./results/results_#1_17042023.csv";


echo Finished extraction!

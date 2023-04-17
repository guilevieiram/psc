#!/bin/bash


echo Extracting models
echo Count before: 
ls ./psc/finals | wc -l;

cd ~;

mkdir pscdata;
cd pscdata;

scp -r guilherme.vieira-manhaes@vanda.polytechnique.fr:~/*.fr .;

mkdir finals results;
mmv "*.fr/psc/whatIs/finals/*.pkl" "./finals/#2_05042023_v3_#1.pkl";
mmv "*.fr/psc/whatIs/results.csv" "./results/results_#1_05042023_v3.csv";


cd ~;
cp -r ./pscdata/finals/*.pkl ./psc/finals/;
cp -r ./pscdata/results/*.csv ./psc/finals/;


echo Count after: 
ls ./psc/finals | wc -l;


echo Finished extraction!

#!/bin/bash

export key=1vb8DgNOWRbkn6n6nDQNk1kBFSmXYQ


export name='data'

keywords="Brain;neurons;Neuronal+activity;Brain+activity;fluorescent+neuronal;Brain+circuit;Brain+background;Abstract=art+networks;Abstract+neurons;Abstract+brain;Brain+signals;Neurons+connections;Brain+connections; Neurons+background;Brain+networks; "  
keywords_array=($(echo $keywords | tr ";" "\n"))

i=0
for keyword in "${keywords_array[@]}"
do
    echo keyword
    curl -H "Authorization: Bearer ${key}" "https://api.openverse.engineering/v1/images/?q='$keyword'" > data/$i.json 
    let "i=i+1"

done


python download_openverse.py
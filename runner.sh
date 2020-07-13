#!/bin/bash

SD=(36)
TD=(64)
RD=(0 )

for DS in ${@}; do
    for i in $(seq 0 0); do
        cp run.template.sh run.${DS}.${i}.sh
        sed -i "s/DS/${DS}/" run.${DS}.${i}.sh
        sed -i "s/SD/${SD[i]}/" run.${DS}.${i}.sh
        sed -i "s/TD/${TD[i]}/" run.${DS}.${i}.sh
        sed -i "s/RD/${RD[i]}/" run.${DS}.${i}.sh
        sbatch run.${DS}.${i}.sh
        rm run.${DS}.${i}.sh
    done
done

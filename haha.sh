#!/bin/bash
while ((0<1))
do
    c=$(nvidia-smi  -i 1 -q --display=MEMORY | grep Free | head -1 | awk '{print $(NF-1)}')
    a=$(nvidia-smi  -i 0 -q --display=MEMORY | grep Free | head -1 | awk '{print $(NF-1)}')
    b=10000
    if [ $a -gt $b ]
    then
        echo 'using gpu 0'
        python train.py -gpu '0'
        break
    fi

    if [ $c -gt $b ]
    then
        echo 'using gpu 1'
        python train.py -gpu '1'
        break
    fi
    sleep 2s
    echo $a
    echo $c
done

#!/bin/bash
tasks=('ordinary' 'user_1' 'user_3' 'user_5' 'product_1' 'product_3' 'product_5')
for task in ${tasks[@]}; do
  nohup ./auto_train.sh ${task} > /home/share/yinxiangkun/nohup/${task}.log 2>&1 &
done
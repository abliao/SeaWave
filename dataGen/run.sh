#!/bin/bash

port=30011
host="127.0.0.1:${port}"

data_info="固定桌子种类"
n_objs=2
handSide="Right"
event='graspTargetObj'
add_info='0402'
output_path="${n_objs}_objs_${event}_${handSide}_${add_info}"
# 定义程序的命令或路径
command1="/data2/liangxiwen/zkd/simulator/Linux-02-20/HARIX_RDKSim/Binaries/Linux/HARIX_RDKSim HARIX_RDKSim -graphicsadapter=5 -port=${port} -RenderOffScreen"

command2="python dataGen.py --host=${host} --output_path=${output_path} --data_info=${data_info} --n_objs=${n_objs} --handSide=${handSide} --event=${event}"

# 定义重启间隔时间（秒）
restart_interval=7200  # 例如，每2小时重启一次

while true; do
    # 启动两个程序
    echo "Starting program 1..."
    $command1 &
    pid1=$!  # 获取程序1的PID
    echo "Program 1 started with PID $pid1"
    sleep 10

    echo "Starting program 2..."
    $command2 > "${output_path}.log" &
    pid2=$!  # 获取程序2的PID
    echo "Program 2 started with PID $pid2"

    # 等待指定的时间
    echo "Sleeping for $restart_interval seconds..."
    sleep $restart_interval

    # 杀掉两个程序
    echo "Killing program 1 (PID: $pid1)..."
    kill $pid1
    echo "Killing program 2 (PID: $pid2)..."
    kill $pid2

    # 等待一小段时间，确保程序已经完全关闭
    sleep 10
done

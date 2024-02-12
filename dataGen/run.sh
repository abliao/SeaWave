#!/bin/bash


# 定义程序的命令或路径
command1="/data2/liangxiwen/zkd/simulator/Linux-01-29/HARIX_RDKSim/Binaries/Linux/HARIX_RDKSim HARIX_RDKSim -graphicsadapter=5 -port=30008 -RenderOffScreen"

command2="python RRTGen.py"

# 定义重启间隔时间（秒）
restart_interval=3600  # 例如，每小时重启一次

while true; do
    # 启动两个程序
    echo "Starting program 1..."
    $command1 &
    pid1=$!  # 获取程序1的PID
    echo "Program 1 started with PID $pid1"
    sleep 10

    echo "Starting program 2..."
    $command2 &
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

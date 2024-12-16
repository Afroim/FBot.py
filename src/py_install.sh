#!/bin/bash

# Обновление списка пакетов (если требуется)
echo "Обновление списка пакетов..."
sudo apt update -y

# Установка Python и pip (если не установлены)
echo "Проверка установки Python и pip..."
if ! command -v python3 &> /dev/null; then
    echo "Устанавливается Python3..."
    sudo apt install python3 -y
fi

if ! command -v pip3 &> /dev/null; then
    echo "Устанавливается pip..."
    sudo apt install python3-pip -y
fi

# Установка необходимых библиотек
echo "Установка Python-библиотек..."
pip3 install numpy deap tabulate pandas

# Проверка установки библиотек
echo "Проверка установленных библиотек..."
pip3 show numpy deap tabulate pandas

echo "Все необходимые библиотеки установлены!"
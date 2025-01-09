# -*- coding: utf-8 -*-
# @Time : 2024/3/12 18:04
# @Author : Weiqi
# @File : train.py.py

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo_Rep.pt")
    model = YOLO("yolo_Rep.yaml")

    model.train(model="yolo_Rep.pt",
                data="my_data.yaml", epochs=300, patience=200,
                imgsz=640)

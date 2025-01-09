# -*- coding: utf-8 -*-
# @Time : 2024/3/18 20:16
# @Author : Weiqi
# @File : predict.py

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best.pt")
    model = YOLO("best.pt")

    model.predict(model="best.pt",
                  source="test_data", save=True)

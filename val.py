# -*- coding: utf-8 -*-
# @Time : 2024/3/18 20:08
# @Author : Weiqi
# @File : val.py

# -*- coding: utf-8 -*-
# @Time : 2024/3/12 18:04
# @Author : Weiqi
# @File : train.py.py

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best.pt")
    model = YOLO("best.pt")

    model.val(model="best.pt",
              data="my_data.yaml", epochs=300, imgsz=640)

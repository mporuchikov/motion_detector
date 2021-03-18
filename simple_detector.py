#!/usr/bin/python3

import cv2
import numpy as np
from datetime import datetime

backSub = cv2.createBackgroundSubtractorMOG2(50, 16, True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    # шаг 1: получить изображение

    _, frame = cap.read()

    # шаг 2: вычитание фона

    fg_mask = backSub.apply(frame)

    # шаг 3: бинаризация маски

    _, mask_thr = cv2.threshold(fg_mask, 100, 255, 0) # с тенями

    # шаг 4: исключение мелкого шума

    kernel_open = np.ones((5,5), np.uint8)
    mask_open = cv2.morphologyEx(mask_thr, cv2.MORPH_OPEN, kernel_open)

    # шаг 5: исключение мелких областей

    kernel_close = np.ones((9,9), np.uint8)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

    # шаг 6: поиск контуров

    _, contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # шаг 7: исключение контуров малой площади

    area_threshold = 100
    contours_sel = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    # шаг 8: расчет площади контуров

    total_area = 0
    for cnt in contours_sel:
        total_area += cv2.contourArea(cnt)
    rel_area = total_area / (frame.shape[0] * frame.shape[1]) * 100

    # шаг 9: проверка движения в кадре

    motion_threshold = 0.5

    if rel_area > motion_threshold:

        # 9.1 отрисовка ограничивающих прямоугольников

        frame_boxes = frame.copy()
        for cnt in contours_sel:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 9.2 отображение даты и времени

        dt = datetime.now()
        dt_image = dt.strftime('%d.%m.%Y %H:%M:%S.%f')[:-3]
        cv2.putText(frame_boxes, dt_image, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 9.3 сохранение файла

        dt_file = dt.strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
        fname_out = 'images/' + dt_file + '.jpg'
        cv2.imwrite(fname_out, frame_boxes)
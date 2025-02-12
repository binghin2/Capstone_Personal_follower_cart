import cv2
import numpy as np
from picamera2 import Picamera2
import threading
import time
from gpiozero import Motor, PWMOutputDevice
from time import sleep

#-----------------------------------------------모터제어-------------------------------------------------
# 1번모터 GPIO26 forward , GPIO19 backward GPIO13 ENA
# 2번모터 GPIO21 forward , GPIO16 backward GPIO12 ENA
# 첫 번째 모터 핀 설정 (GPIO 26번 forward 물리적 37번, GPIO 19번 backward 물리적 35번, GPIO 13번 ENA 물리적 33번)
motor1 = Motor(forward=26, backward=19)
motor1_ENA = PWMOutputDevice(13)  # GPIO 13번 핀 (물리적 핀 번호 33번)

# 두 번째 모터 핀 설정 (GPIO 21번 forward 물리적 40번, GPIO 16번 backward 물리적 36번, GPIO 12번 ENA 물리적 32번)
motor2 = Motor(forward=21, backward=16)
motor2_ENA = PWMOutputDevice(12)  # GPIO 12번 핀 (물리적 핀 번호 32번)
#---------------------------------------------------------------------------------------------------------

#-----------------------------------------------YOLO-----------------------------------------------------
# YOLOv4-tiny 모델과 구성 파일의 경로
weights_path = "/home/pi/yolov4/yolov4-tiny.weights"
config_path = "/home/pi/yolov4/yolov4-tiny.cfg"

# 클래스 이름 파일 경로
class_names_path = "/home/pi/yolov4/coco.names"

# 클래스 이름 로드
with open(class_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------VISION 설정-------------------------------------------------
# 네트워크 로드
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# PiCamera2 설정
camera = Picamera2(camera_num=0)
camera.configure(camera.create_video_configuration(main={"size": (1280, 960)}))
camera.start()

# 초점 거리와 실제 사람의 너비 (cm)
focal_length_px = 1056.32  # 위 계산의 결과
real_person_width = 50  # IMX219 모델의 실제 사람 너비

# NMS 임계값 설정
score_threshold = 0.5
nms_threshold = 0.4  # 이 값을 조정하여 실험해보세요

# 바운딩 박스 크기 조정 비율 (1.0은 원래 크기, 1.1은 10% 확장, 0.9는 10% 축소)
bbox_adjustment_factor = 0.8
#----------------------------------------------------------------------------------------------------------

#---------------------------------------------이동 방향 추정------------------------------------------------
# 이동 방향 추정을 위한 변수들
prev_center_x = None
movement_threshold = 5  # 픽셀 이동 임계값
direction_history = []
history_length = 5  # 이동 평균을 계산할 프레임 수

def calculate_distance(focal_length, real_width, width_in_frame):
    # 거리 계산 공식
    return (real_width * focal_length) / width_in_frame
#-----------------------------------------------------------------------------------------------------------

#--------------------------------------------객체 검출 결과 설정----------------------------------------------
def adjust_bbox(x, y, w, h, adjustment_factor):
    # 바운딩 박스 크기 조정
    new_w = int(w * adjustment_factor)
    new_h = int(h * adjustment_factor)
    new_x = x - (new_w - w) // 2
    new_y = y - (new_h - h) // 2
    return new_x, new_y, new_w, new_h

def process_frame(frame):
    global prev_center_x, direction_history

    height, width, channels = frame.shape

    # 네트워크에 프레임 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 검출 정보 초기화
    class_ids = []
    confidences = []
    boxes = []

    # 네트워크 출력 분석
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > score_threshold and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

    # 검출 결과 처리
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            x, y, w, h = adjust_bbox(x, y, w, h, bbox_adjustment_factor)
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            distance = calculate_distance(focal_length_px, real_person_width, w)

            # 바운딩 박스와 거리 정보 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f} Distance: {distance:.2f} cm", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 이동 방향 추정 (좌우만)
            center_x = x + w // 2
            if prev_center_x is not None:
                dx = center_x - prev_center_x

                if abs(dx) > movement_threshold:
                    if dx > 0:
                        direction = "오른쪽으로 이동"
                    else:
                        direction = "왼쪽으로 이동"
                    direction_history.append(direction)
                    if len(direction_history) > history_length:
                        direction_history.pop(0)

                    # 이동 방향의 평균 계산
                    if direction_history.count("오른쪽으로 이동") > direction_history.count("왼쪽으로 이동"):
                        final_direction = "오른쪽으로 이동"
                    else:
                        final_direction = "왼쪽으로 이동"

                    cv2.putText(frame, final_direction, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(final_direction)
                    print(distance)
                    
                    # 이동에 따른 모터 제어 추가
                    if distance >= 100:
                        if final_direction == "왼쪽으로 이동":
                            motor1_ENA.value = 0.2
                            motor2_ENA.value = 1.0
                            motor1.backward()
                            motor2.forward()
                        elif final_direction == "오른쪽으로 이동":
                            motor1_ENA.value = 1.0
                            motor2_ENA.value = 0.2
                            motor1.forward()
                            motor2.backward()
                        else:
                            motor1_ENA.value = 0.8
                            motor2_ENA.value = 0.8
                            motor1.forward()
                            motor2.forward()
                    else:
                        motor1.stop()
                        motor2.stop()

            # 현재 중심 좌표를 이전 중심 좌표로 업데이트
            prev_center_x = center_x

    return frame

def capture_and_process():
    frame_count = 0
    while True:
        start_time = time.time()
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # 프레임 스킵 (예: 매 2번째 프레임만 처리)
        if frame_count % 2 == 0:
            processed_frame = process_frame(frame)

            # 결과 프레임 표시
            cv2.imshow("Frame", processed_frame)

        frame_count += 1

        # FPS 계산 및 표시
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 별도의 스레드에서 캡처 및 처리 시작
capture_thread = threading.Thread(target=capture_and_process)
capture_thread.start()

# 메인 스레드가 종료되지 않도록 대기
capture_thread.join()

# 자원 해제
camera.stop()
cv2.destroyAllWindows()

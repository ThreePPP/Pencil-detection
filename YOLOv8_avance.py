from ultralytics import YOLO
import cv2
import math

# [cite_start]1. เปิดกล้อง Webcam (0 คือกล้องตัวแรก) [cite: 482]
myvideo = cv2.VideoCapture(0)

# [cite_start]2. โหลดโมเดล YOLOv8 nano [cite: 485]
model = YOLO("yolov8n.pt")

# [cite_start]3. ดึงชื่อ Class ของวัตถุทั้งหมด (เช่น 'person', 'car') [cite: 489]
OBJnames = model.names

while True:
    # อ่านภาพจากกล้อง
    success, img = myvideo.read()
    
    # ถ้าอ่านภาพไม่สำเร็จ (เช่น กล้องหลุด) ให้ข้ามรอบนี้
    if not success:
        continue

    # [cite_start]ส่งภาพให้ YOLO ตรวจจับ (stream=True ช่วยลดเมมโมรี่) [cite: 498]
    results = model(img, stream=True)

    # วนลูปดึงข้อมูลผลลัพธ์
    for region in results:
        boxes = region.boxes
        for box in boxes:
            # [cite_start]--- หาพิกัดกรอบสี่เหลี่ยม (Bounding Box) --- [cite: 511-513]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # วาดกรอบสี่เหลี่ยมลงบนภาพ
            # [cite_start]สี (255, 0, 255) คือสีม่วง, ความหนา 3 [cite: 517]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # [cite_start]--- คำนวณค่าความมั่นใจ (Confidence Score) --- [cite: 521]
            confidence = math.ceil((box.conf[0] * 100)) / 100
            
            # [cite_start]--- หาชื่อ Class ของวัตถุ --- [cite: 528]
            cls = int(box.cls[0])
            currentClass = OBJnames[cls]

            # เตรียมข้อความที่จะแสดงบนจอ
            text_display = f'{currentClass} {confidence}'
            
            # [cite_start]กำหนดจุดวางข้อความและรูปแบบฟอนต์ [cite: 535-552]
            org = (x1, y1 - 10) # วางเหนือกรอบนิดหน่อย
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0) # สีน้ำเงิน
            thickness = 2
            
            cv2.putText(img, text_display, org, font, fontScale, color, thickness)

    # [cite_start]แสดงผลภาพทางหน้าต่างชื่อ 'My Webcam' [cite: 553]
    cv2.imshow('My Webcam', img)

    # [cite_start]กดปุ่ม 'q' เพื่อออกจากโปรแกรม [cite: 554-555]
    if cv2.waitKey(1) == ord('q'):
        break

# คืนทรัพยากรกล้องและปิดหน้าต่าง
myvideo.release()
cv2.destroyAllWindows()
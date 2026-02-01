import ultralytics
import time
from ultralytics import YOLO

# ฟังก์ชันตรวจสอบความพร้อมของระบบ [cite: 290]
print(ultralytics.checks())

# 1. ฟังก์ชันตรวจจับผ่านกล้อง Webcam [cite: 307]
def detectWebCam():
    model = YOLO('runs/detect/pencil_detection5/weights/best.pt') # โหลดโมเดลที่เทรนแล้ว
    # source=0 คือกล้อง Webcam, show=True คือแสดงผลทันที
    results = model.track(source=0, show=True)

# 2. ฟังก์ชันตรวจจับผ่านวิดีโอ YouTube [cite: 260]
def detectVideoURL():
    model = YOLO('runs/detect/pencil_detection5/weights/best.pt')
    # ตัวอย่างวิดีโอจากเอกสาร
    results = model.track(source="https://youtu.be/Ysr56eUoniM?t=5", show=True)

# 3. ฟังก์ชันตรวจจับแบบ Segmentation (แยกพื้นที่วัตถุ) [cite: 270]
def detectsegment():
    model = YOLO("yolov8n-seg.pt") # ใช้โมเดลสำหรับ Segmentation [cite: 275]
    results = model.track(source="https://youtu.be/Ysr56eUoniM?t=5", show=True)

# 4. ฟังก์ชันตรวจจับจากรูปภาพ [cite: 284]
def detectImage():
    model = YOLO('runs/detect/pencil_detection5/weights/best.pt')
    results = model.track(source="imageTest.jpg", show=True) # ต้องมีไฟล์ภาพชื่อ imageTest.jpg ในโปรเจกต์
    time.sleep(5)

# --- ส่วนเรียกใช้งานฟังก์ชัน (เลือกเปิดทีละอัน) --- [cite: 292-298]
if __name__ == '__main__':
    detectWebCam()       # ทดสอบกล้อง Webcam
    # detectVideoURL()   # ทดสอบวิดีโอ
    # detectsegment()    # ทดสอบ Segmentation
    # detectImage()      # ทดสอบรูปภาพ
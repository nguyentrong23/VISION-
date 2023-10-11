import cv2
import requests

# viết thêm chương trình truy cập vào noi dung cua qr code
qrc = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (860, 620))
    if ret:
        ret_qr,decoded_info, points,_ = qrc.detectAndDecodeMulti(frame)
        if ret_qr:
            for s,p in zip(decoded_info,points):
                if s:
                      cv2.putText(frame, s, (100,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,50),1, cv2.LINE_AA)
                else:
                    color = (0,0,255)
                cv2.polylines(frame, [p.astype(int)],True, (0, 255, 0), 2)

                try:
                    response = requests.get(s)

                    if response.status_code == 200:
                        print("Truy cập thành công!")
                        content = response.content
                    else:
                        print("Truy cập không thành công. Mã trạng thái:", response.status_code)

                except requests.exceptions.RequestException as e:
                    print(f"Có lỗi xảy ra: {e}")

        cv2.imshow('QR code', frame)

        # Thoát vòng lặp nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
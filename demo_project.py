import cv2
import hand as hd
fingers = []
finger = []
cap = cv2.VideoCapture('http://192.168.1.102:8080/video')
detector = hd.handDetector()
fingerid = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (860, 620))
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        print("da phat hien ban tay")
        if (lmList[fingerid[0]][1] > lmList[fingerid[0] - 1][1]) & (
                lmList[fingerid[1]][2] > lmList[fingerid[1] - 2][2]) & (
                lmList[fingerid[4]][2] < lmList[fingerid[4] - 2][2]):
            cv2.putText(frame, "OK", (30, 390), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 5)

        if (lmList[fingerid[0]][2] < lmList[fingerid[0] - 2][2]) & (
                lmList[fingerid[1]][1] < lmList[fingerid[1] - 2][1]) & (
                lmList[fingerid[4]][1] < lmList[fingerid[4] - 2][1]):
            cv2.putText(frame, "LIKE", (30, 390), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 5)

        if (lmList[fingerid[0]][2] < lmList[fingerid[0] - 2][2]) & (
                lmList[fingerid[4]][1] > lmList[fingerid[4] - 2][1]) & (
                lmList[fingerid[1]][1] < lmList[fingerid[1] - 2][1]):
            if (lmList[fingerid[4]][1] > lmList[fingerid[4] - 2][1]):
                cv2.putText(frame, "CALL ME ", (30, 390), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
        if (lmList[fingerid[0]][2] < lmList[fingerid[0] - 2][2]) & (
                lmList[fingerid[1]][2] < lmList[fingerid[1] - 2][2]) & (
                lmList[fingerid[2]][2] > lmList[fingerid[2] - 2][2]) & (
                lmList[fingerid[3]][2] > lmList[fingerid[3] - 2][2]) & (
                lmList[fingerid[4]][2] > lmList[fingerid[4] - 2][2]):
            cv2.putText(frame, "CHECK ", (30, 390), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
    else:
        print("k tim thay ban tay nguoi")
    cv2.imshow("testcode", frame)

    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ

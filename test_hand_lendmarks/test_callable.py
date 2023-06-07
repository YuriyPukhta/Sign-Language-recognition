from callable import ASLDetector
import cv2


def process(frame, letter):
    cv2.imshow("Frame", frame)
    print(letter)


detector = ASLDetector()
detector.launch(process)

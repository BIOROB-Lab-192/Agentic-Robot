import threading
import time

import cv2
from vidgear.gears import CamGear, WriteGear


class Webcam:
    def __init__(self, camera_id, resolution):

        self.resolution = resolution
        self.camera_id = camera_id

        options = {
            "CAP_PROP_FRAME_WIDTH": self.resolution[0],
            "CAP_PROP_FRAME_HEIGHT": self.resolution[1],
            "CAP_PROP_FPS": 60,
        }
        self.cam = CamGear(source=self.camera_id, logging=False, **options).start()

        self.running = True

    def get_frame(self):
        frame = self.cam.read()
        return frame

    def _display_loop(self):
        while self.running:
            frame = self.get_frame()

            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def _save_vid_loop(self):
        pass

    def start_display(self):
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

    def stop_webcam(self):
        self.running = False
        cv2.destroyAllWindows()
        self.cam.stop()


if __name__ == "__main__":
    cam = Webcam(0, (1920, 1080))
    cam.start_display()
    for i in range(10):
        print(i)
        time.sleep(1)
    cam.stop_webcam()

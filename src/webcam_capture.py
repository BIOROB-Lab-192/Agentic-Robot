from vidgear.gears import CamGear, WriteGear
import cv2

class Webcam:
    def __init__(self, camera_id, resolution):
        
        self.resolution = resolution
        self.camera_id = camera_id
        
        options = {
            "CAP_PROP_FRAME_WIDTH": self.resolution[0],
            "CAP_PROP_FRAME_HEIGHT": self.resolution[1],
            "CAP_PROP_FPS": 60
        }
        self.cam = CamGear(source=self.camera_id, logging=False, **options).start()
        
        
    def stop_webcam(self):
        cv2.destroyAllWindows()
        self.cam.stop()
        
    def show_vid(self):
        while True:
            frame = self.cam.read()
            
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF  
            if key == ord("q"):
                break

if __name__ == "__main__":
    cam = Webcam(0, (1024, 1024))
    cam.show_vid()
    cam.stop_webcam()
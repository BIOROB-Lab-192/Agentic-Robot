import src.depth_camera as depthcam
import src.LLM_interface as llm
import src.tools as tools
import src.webcam_capture as webcam

webcam_res = (1920, 1080)
depth_res = (1280, 720)


def main():
    # Auto-load calibration_out/calibration.json if present; otherwise run in raw camera coords.
    tools.load_calibration("calibration_out/calibration.json")

    interface = llm.LLMinterface(
        "models/Qwen3.5-4B-Q4_K_M.gguf", tools.build_tools(webcam_res, depth_res)
    )
    cam = webcam.Webcam(0, webcam_res)
    depth_cam = depthcam.RealSense(depth_res)
    try:
        while True:
            interface.get_text()
            interface.send_message_with_tools(cam, depth_cam)
            interface.print_message()
    except Exception as e:
        print(e)
    finally:
        print("exiting")
        depth_cam.stop()
        cam.stop_webcam()
        print("done")


if __name__ == "__main__":
    main()

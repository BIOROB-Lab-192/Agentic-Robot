import src.depth_camera as depthcam
import src.LLM_interface as llm
import src.tools as tools
import src.webcam_capture as webcam


def main():
    interface = llm.LLMinterface("models/Qwen3.5-4B-Q4_K_M.gguf", tools.tool_json_list)
    cam = webcam.Webcam(0, (1920, 1080))
    depth_cam = depthcam.RealSense()
    try:
        while True:
            interface.get_text()
            interface.send_message_with_tools(cam, depth_cam)
            interface.print_message()
    except Exception as e:
        print(e)
    finally:
        cam.stop_webcam()
        depth_cam.stop()


if __name__ == "__main__":
    main()

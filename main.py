import src.LLM_interface as llm
import src.webcam_capture as webcam

def main():
    interface = llm.LLMinterface("models/Qwen3.5-4B-Q4_K_M.gguf")
    cam = webcam.Webcam(0, (1920,1080))
    while True:
        interface.get_text()
        interface.send_message()
        interface.print_message()
    

if __name__ == "__main__":
    main()
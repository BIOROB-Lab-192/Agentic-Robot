Project to approximate the functionality of a Vision-Language-Action model for robotics. Using an vision and language models to interpret instructions, identify objects, their coordinates and the required motion. The final approach is not fully determined and so this project is currently exploratory.

There are two possible approaches, a unified VL model such as QWEN 3.5 to process the prompt, identify and return coordinates in the image. Alternatively a vision model such as YOLO26 passes scene information to a language model that interprets the information with the prompt to determine the task.

The final coordinates will be passed off to the robot SDK to perform the task action directly.
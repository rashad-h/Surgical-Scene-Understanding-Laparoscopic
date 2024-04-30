import cv2
import numpy as np
from loss_function.losses import *
from tensorflow.keras.models import load_model
import time

# Load the segmentation model
model_name = 'AUGMENTED-WEIGHTED-CUSTOM-CROSS-ENTROPY-60-epochepoch'
model = load_model(model_name + ".keras", custom_objects={"dice_metric": dice_metric,
                                                                             "dice_metric_0": dice_metric_0,
                                                                             "dice_metric_1": dice_metric_1,
                                                                             "dice_metric_2": dice_metric_2,
                                                                             "dice_metric_3": dice_metric_3,
                                                                             "dice_metric_4": dice_metric_4,
                                                                             "dice_metric_5": dice_metric_5,
                                                                             "dice_metric_6": dice_metric_6,
                                                                             "dice_metric_7": dice_metric_7,
                                                                             "dice_metric_8": dice_metric_8,
                                                                             "dice_metric_9": dice_metric_9,
                                                                             "loss": 1,
                                                                             "categorical_focal_loss_fixed": 1})
                                                                        
print("model loaded successfuly!")


colors = [
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 0, 0],    # Maroon
    [0, 128, 0],    # Green (Forest)
    [100, 100, 100],     # Navy

]

# Function to preprocess frame and apply segmentation mask
def process_frame(frame, frame_number):
    print(frame_number)


    frame = cv2.resize(frame, (256, 256))

    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mask = model.predict(np.expand_dims(new_frame, axis=0))[0]
    mask = np.argmax(mask, axis=-1)
    
    # Overlay mask on frame
    overlay = np.zeros_like(frame)
    for i in range(1, 10):
        overlay[mask == i] = colors[i-1]
    

    overlay = cv2.resize(overlay, (1920, 1080))
    return overlay

# Function to process video
def process_video(input_path, output_path):
    print("start processing video")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    frame_number = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, frame_number)
        frame_number += 1
        final_result = cv2.addWeighted(frame, 1, processed_frame, 0.8, 0)
        out.write(final_result)
        
            
        cv2.imshow('Processed Video', final_result)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q') or frame_number == 2:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("finished")

# Example usage
input_video_path = r'C:\Users\Rashad\Desktop\dissertation\Surgical-Scene-Understanding-Laparoscopic\videos\01.mp4'
output_video_path = f"C:\\Users\\Rashad\\Desktop\\dissertation\\Surgical-Scene-Understanding-Laparoscopic\\masked_vid\\{model_name}01sdfsdf.mp4"
process_video(input_video_path, output_video_path)

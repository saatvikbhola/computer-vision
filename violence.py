import gradio as gr
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import shutil
import os
import tempfile

# ---------------------------
# ENVIRONMENT CLEANUP
# ---------------------------
# Automatically clear Gradio cache to prevent output mismatch errors
try:
    shutil.rmtree(os.path.join(tempfile.gettempdir(), "gradio"))
    print("🧹 Cleared old Gradio cache.")
except FileNotFoundError:
    pass

# ---------------------------
# CONSTANTS
# ---------------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# Global model variable
model = None


# ---------------------------
# MODEL LOADING
# ---------------------------
def load_violence_model(model_file):
    """Load the pre-trained violence detection model"""
    global model
    try:
        if model_file is None:
            return "❌ Please upload a model file first!"

        model_path = model_file.name
        print(f"🔄 Loading model from: {model_path}")

        model = load_model(model_path, compile=False)

        if model is not None:
            print("✅ Model loaded successfully!")
            return f"✅ Model loaded successfully! Ready for video analysis."
        else:
            return "❌ Model loading failed"
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return error_msg


# ---------------------------
# FRAME EXTRACTION
# ---------------------------
def extract_frames(video_path):
    """Extract frames from video for prediction"""
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list


# ---------------------------
# PREDICTION
# ---------------------------
def predict_on_frames(frames_list):
    """Make prediction on a sequence of frames"""
    if len(frames_list) < SEQUENCE_LENGTH:
        return None, None

    predicted_probs = model.predict(np.expand_dims(frames_list, axis=0), verbose=0)[0]
    label_index = np.argmax(predicted_probs)
    predicted_class = CLASSES_LIST[label_index]
    confidence = predicted_probs[label_index]

    return predicted_class, confidence


# ---------------------------
# VIDEO PROCESSING
# ---------------------------
def process_video_optimized(video_file):
    """Process the uploaded video and run violence detection"""
    global model
    if model is None:
        print("⚠️ Model not loaded")
        return "❌ Please load the model first!", None, None, None, None, None

    try:
        video_path = video_file.name
        print(f"📹 Processing video: {video_path}")

        video_reader = cv2.VideoCapture(video_path)
        frames_queue = deque(maxlen=SEQUENCE_LENGTH)

        fps = video_reader.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        results, sample_frames, frame_predictions = [], [], []
        frame_count = 0
        prediction_interval = 8

        video_info = f"""
        📊 **Video Information:**
        • Duration: {duration:.2f} seconds  
        • Total Frames: {total_frames}  
        • Frame Rate: {fps:.2f} FPS  
        • Resolution: {width}x{height}
        """

        print("🎬 Starting video analysis...")

        while video_reader.isOpened():
            success, frame = video_reader.read()
            if not success:
                break

            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_queue.append(normalized_frame)

            current_prediction = None
            current_confidence = 0

            if len(frames_queue) == SEQUENCE_LENGTH and frame_count % prediction_interval == 0:
                predicted_class, confidence = predict_on_frames(list(frames_queue))
                if predicted_class:
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    results.append({
                        'timestamp': timestamp,
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'is_violence': predicted_class == "Violence"
                    })
                    current_prediction = predicted_class
                    current_confidence = confidence

                    # Capture a few sample frames for visualization
                    if frame_count % max(int(3 * fps), 90) == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        sample_frames.append({
                            'frame': frame_rgb,
                            'timestamp': timestamp,
                            'prediction': predicted_class,
                            'confidence': confidence
                        })

            # Store frame with prediction for annotated video
            frame_predictions.append({
                'frame': frame.copy(),
                'prediction': current_prediction,
                'confidence': current_confidence,
                'frame_count': frame_count,
                'timestamp': frame_count / fps if fps > 0 else frame_count
            })

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"📊 Processed {frame_count}/{total_frames} frames")

        video_reader.release()

        if not results:
            print("⚠️ No results generated.")
            return "❌ No results generated. Video might be too short.", None, None, None, None, None

        print("✅ Video analysis completed! Returning 6 outputs.")
        return video_info, results, sample_frames, frame_predictions, duration, total_frames

    except Exception as e:
        error_msg = f"❌ Error processing video: {str(e)}"
        print(error_msg)
        print("Returning 6 values (error path)")
        return error_msg, None, None, None, None, None


# ---------------------------
# ANNOTATED VIDEO CREATION
# ---------------------------
def create_annotated_video(frame_predictions, original_video_path):
    """Create a new video with violence detection annotations"""
    if not frame_predictions:
        return None
    
    try:
        # Get video properties from first frame
        first_frame = frame_predictions[0]['frame']
        height, width = first_frame.shape[:2]
        fps = 30  # Default FPS, you can extract from original video if needed
        
        # Create temporary file for output video
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_output.name
        temp_output.close()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎥 Creating annotated video with {len(frame_predictions)} frames...")
        
        for i, frame_data in enumerate(frame_predictions):
            frame = frame_data['frame']
            prediction = frame_data['prediction']
            confidence = frame_data['confidence']
            
            # Convert BGR to RGB for processing (we'll convert back for writing)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add timestamp
            timestamp = frame_data['timestamp']
            cv2.putText(frame_rgb, f"Time: {timestamp:.2f}s", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add prediction info if available
            if prediction:
                color = (255, 0, 0) if prediction == "Violence" else (0, 255, 0)
                status_text = f"{prediction}: {confidence:.3f}"
                
                # Add background for text
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(frame_rgb, (20, 60), (20 + text_size[0] + 10, 60 + text_size[1] + 10), 
                             (0, 0, 0), -1)
                
                # Add prediction text
                cv2.putText(frame_rgb, status_text, (25, 60 + text_size[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Add bounding box around frame if violence detected
                if prediction == "Violence":
                    cv2.rectangle(frame_rgb, (10, 10), (width-10, height-10), color, 8)
            
            # Add frame counter
            cv2.putText(frame_rgb, f"Frame: {i}/{len(frame_predictions)}", (width-300, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Convert back to BGR for writing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if i % 100 == 0:
                print(f"📹 Writing frame {i}/{len(frame_predictions)}")
        
        out.release()
        print(f"✅ Annotated video created: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error creating annotated video: {str(e)}")
        return None


# ---------------------------
# VISUALIZATION
# ---------------------------
def create_visualization(results, duration):
    """Generate matplotlib visualization for confidence and violence probability"""
    if not results:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    timestamps = [r['timestamp'] for r in results]
    confidences = [r['confidence'] for r in results]
    is_violence = [r['is_violence'] for r in results]

    colors = ['red' if v else 'green' for v in is_violence]
    ax1.scatter(timestamps, confidences, c=colors, alpha=0.6, s=20)
    ax1.plot(timestamps, confidences, 'b-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Violence Detection Confidence Over Time')
    ax1.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color='blue', alpha=0.3, linewidth=2, label='Confidence Trend'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Violence'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Non-Violence')
    ]
    ax1.legend(handles=legend)

    violence_probs = [r['confidence'] if r['is_violence'] else 1 - r['confidence'] for r in results]
    ax2.plot(timestamps, violence_probs, 'r-', linewidth=2, alpha=0.8)
    ax2.fill_between(timestamps, violence_probs, alpha=0.3, color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Violence Probability')
    ax2.set_title('Violence Probability Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


# ---------------------------
# RESULT ANALYSIS
# ---------------------------
def analyze_results(video_info, results, sample_frames, frame_predictions, duration, total_frames, video_file):
    """Generate final report, visualization, and CSV"""
    if results is None:
        return video_info, None, None, None, None

    violence_frames = sum(1 for r in results if r['is_violence'])
    total_preds = len(results)
    violence_pct = (violence_frames / total_preds) * 100 if total_preds else 0
    avg_conf = np.mean([r['confidence'] for r in results])

    if violence_pct > 50:
        verdict, msg = "🚨 VIOLENCE DETECTED", "Significant violent activity detected."
    elif violence_pct > 20:
        verdict, msg = "⚠️ SUSPICIOUS ACTIVITY", "Possible violent activity observed."
    else:
        verdict, msg = "✅ NO VIOLENCE DETECTED", "No significant violent activity detected."

    summary = f"""
    📋 **Analysis Results:**
    **{verdict}**
    
    **Statistics:**
    • Total Predictions: {total_preds}  
    • Violence Detected: {violence_frames}  
    • Violence Percentage: {violence_pct:.1f}%  
    • Average Confidence: {avg_conf:.3f}  
    
    **Verdict:** {msg}
    """

    # Create visualization
    fig = create_visualization(results, duration)
    
    # Create gallery
    gallery = [s['frame'] for s in sample_frames[:6]] if sample_frames else []
    
    # Create CSV
    csv_data = pd.DataFrame(results).to_csv(index=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        f.write(csv_data)
        csv_path = f.name
    
    # Create annotated video
    annotated_video_path = None
    if frame_predictions and video_file:
        annotated_video_path = create_annotated_video(frame_predictions, video_file.name)

    return summary, fig, gallery, csv_path, annotated_video_path


# ---------------------------
# GRADIO INTERFACE
# ---------------------------
def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Violence Detection System") as demo:
        gr.Markdown("# 🚨 Violence Detection System\nUpload your trained model and a video to detect violent activities using AI.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Upload Model")
                model_upload = gr.File(label="Upload Model File (.keras)", file_types=[".keras"], type="filepath")
                load_model_btn = gr.Button("🔧 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", value="⏳ Please upload and load model file", interactive=False)

                gr.Markdown("### Step 2: Upload Video")
                video_input = gr.File(label="Upload Video File", file_types=[".mp4", ".avi", ".mov", ".mkv"], type="filepath")
                analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")

            with gr.Column(scale=2):
                video_info = gr.Markdown()
                results_summary = gr.Markdown()
                visualization = gr.Plot()
                sample_gallery = gr.Gallery(label="Sample Frames", columns=3, object_fit="contain")
                
                # New section for annotated video
                with gr.Accordion("🎬 Annotated Video Playback", open=True):
                    gr.Markdown("Watch the video with real-time violence detection annotations:")
                    annotated_video = gr.Video(label="Annotated Video with Detection Results")
                
                csv_download = gr.File(label="📥 Download Results CSV")

        # State variables
        video_results = gr.State()
        sample_frames = gr.State()
        frame_predictions_state = gr.State()
        duration_state = gr.State()
        total_frames_state = gr.State()
        original_video_state = gr.State()

        load_model_btn.click(load_violence_model, inputs=model_upload, outputs=model_status)

        analyze_btn.click(
            fn=process_video_optimized,
            inputs=video_input,
            outputs=[video_info, video_results, sample_frames, frame_predictions_state, duration_state, total_frames_state]
        ).then(
            fn=analyze_results,
            inputs=[video_info, video_results, sample_frames, frame_predictions_state, duration_state, total_frames_state, video_input],
            outputs=[results_summary, visualization, sample_gallery, csv_download, annotated_video]
        )

        with gr.Accordion("ℹ️ Instructions", open=False):
            gr.Markdown("""
            ## 🎬 Annotated Video Features:
            
            The annotated video includes:
            - **Real-time predictions**: Violence/Non-Violence labels with confidence scores
            - **Color-coded alerts**: Red border and text for violence, green for non-violence
            - **Timestamps**: Current time in the video
            - **Frame counter**: Progress through the video
            
            ## 📊 Analysis Outputs:
            1. **Summary Report**: Overall verdict and statistics
            2. **Visualization Charts**: Confidence and probability trends over time
            3. **Sample Frames**: Key frames with detection results
            4. **CSV Download**: Complete timeline data for further analysis
            5. **Annotated Video**: Video playback with real-time annotations
            
            ## 🔧 How to Use:
            1. Upload your `.keras` model file  
            2. Load the model using the button  
            3. Upload a video file (MP4, AVI, MOV, MKV)  
            4. Click **Analyze Video** to process  
            5. Watch the annotated video and review all results
            """)

    return demo


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Violence Detection System...")
    print("📋 Ensure you have:")
    print("   - Your trained .keras model file")
    print("   - A video file to analyze")
    print("\n🌐 The app will open shortly...")

    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
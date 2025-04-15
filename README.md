# Traffic Object Detection & Counting

This project implements a traffic detection and counting system using camera feeds from highways. It automatically detects vehicles in different lanes and counts them over time. The solution leverages Ultralytics' state-of-the-art object detection models for high-accuracy recognition and tracking.

## Features

- **Real-time Vehicle Detection:** Utilize pre-trained YOLO models to detect cars, trucks, and bus.
- **Lane/Plane-based Counting:** Count vehicles in each designated lane or plane.
- **Highway Traffic Analytics:** Generate counts and statistics for traffic analysis.
- **Customizable Pipeline:** Adapt settings and parameters based on camera specifications and environmental conditions.

## Getting Started

### Demo

Experience real-time traffic analysis ! Watch the dynamic demonstration below where our system detects and counts vehicles effortlessly in motion. scenarios.

![Dynamic Traffic Analysis](demo/vehicle_count.gif)

### Prerequisites

- Python 3.11+
- Ultralytics YOLO (install via pip)
- OpenCV for video processing
- Required dependencies as listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/quan23w Traffic_Object_Detection.git
    ```
2. Navigate to the project directory:
    ```
    cd Traffic_Object_Detection
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### Configuration

The counting regions can be modified in ```traffic_count.py``` by changing the ```region_points``` list. Each region is defined by a list of [x, y] coordinates forming a polygon.

## Usage

To run the detection and counting system, execute:
    ```
    python main/traffic_count.py
    ```
The script processes the input video and generates an output video with vehicle detections and counts.



## Project Structure

```markdown
Traffic_Object_Detection/
├── demo/
│   └── vehicle_count.mp4      # Demo output video with vehicle 
├── main/
│   └── traffic_count.py       # Main application script
├── sample/
│   └── video.mp4              # Input video for processing
└── model/
    └── yolo11n.pt             # YOLOv11 model weights
```

## References

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## Contact

For questions or suggestions, please open an issue on GitHub.

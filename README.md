# Advanced AI Safety Monitoring with Fire Violence and Weapon Detection

Real-time AI-powered safety monitoring system capable of detecting:

* Fire

* Violent Behavior

* Weapons (Knife, Scissors)

* People

* Other Objects (Optional Info Mode)

Built using:

* Python

* YOLOv8 (from Ultralytics)

* OpenCV

* Threaded Text-to-Speech Alerts

# Features

**1.Fire Detection**

- Computer vision-based fire detection using HSV color segmentation

- Flicker pattern analysis for realistic fire behavior detection

- Bounding box localization

- Emergency alert triggering

**2.Violence Detection**

- Motion-based violence detection

- Frame-difference spike detection

- Sudden motion variance analysis

- Real-time violent behavior alerts

**3.Weapon Detection**

* Powered by YOLOv8 object detection

* Detects:Knife,Scissors

* Automatic emergency classification

**4.Smart Alert System**

- Multi-frame validation before triggering alerts

- Alert cooldown system

- Emergency vs Informational alerts

- On-screen alert dashboard

**5.Voice Alert System**

- Threaded Text-to-Speech engine

- Non-blocking voice notifications

- Toggle ON/OFF during runtime

# Interactive Controls

| Key     | Action              |

| ------- | ------------------- |

| Q / ESC | Quit                |

| A       | Acknowledge Alerts  |

| V       | Toggle Voice        |

| I       | Toggle Info Mode    |

| S       | Save Screenshot     |

| 1       | Test Fire Alert     |

| 2       | Test Violence Alert |

# System Architecture

The system consists of three main detection modules:

**1️) Object Detection**

- Uses YOLOv8n

- Real-time inference

- Configurable confidence threshold

**2️) Fire Detection Module**

- HSV color masking (Red, Yellow, Orange)

- Flicker pattern analysis

- Contour-based bounding box detection

**3️) Violence Detection Module**

- Frame differencing

- Motion spike detection

- Statistical variance analysis

# Installation

**Clone the Repository**

git clone https://github.com/your-username/advanced-ai-safety-monitor.git

cd advanced-ai-safety-monitor

**Install Dependencies**

pip install -r requirements.txt

Or manually install:

pip install ultralytics opencv-python torch torchvision numpy pyttsx3 pillow

**Download YOLO Weights**

The system uses:

yolov8n.pt

Optional: yolov8n-fire.pt

Optional: yolov8n-pose.pt

Place them in the project directory.

▶️ Run the System

python v2.py

The camera will start automatically.

# Configuration

Inside the AdvancedSafetyMonitor class:

self.config = {

    'width': 960,
    
    'height': 540,
    
    'yolo_confidence': 0.5,
    
    'alert_frames': 3
    
}

You can adjust:

- Resolution

- Detection sensitivity

- Alert triggering frames

# Alert Classification

| Class              | Type          |

| ------------------ | ------------- |

| FIRE_EMERGENCY     | Critical      |

| VIOLENCE_EMERGENCY | Critical      |

| WEAPON             | Critical      |

| PERSON             | Informational |

| WATER BOTTLE       | Informational |

| INFO               | General       |

# Performance Optimizations

- Inference runs every 2 frames

- Cached detections

- Threaded TTS engine

- Frame resizing

- Alert cooldown logic

# Future Improvements

* Custom-trained fire detection model

* Deep-learning based violence classifier

* Web dashboard

* Cloud alert integration

* Email/SMS notifications

* Multi-camera support

* Edge deployment optimization

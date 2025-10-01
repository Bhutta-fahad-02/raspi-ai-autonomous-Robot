# raspi-ai-autonomous-Robot


This project is a miniature robotics platform designed to demonstrate the core capabilities of autonomous vehicles in a low-cost, portable form. Built on a Raspberry Pi with LiDAR, ultrasonic sensors, and camera vision, it integrates mapping, planning, perception, and control into a full-stack robotics system.

**Capabilities:**

_Mapping & Localization: _
    - Builds a real-time occupancy grid from LiDAR scans (Bresenham raytracing). 
    - Maintains robot pose with dead-reckoning motion model.
    
_Autonomous Exploration: _
    - Detects frontiers (boundaries between explored and unknown space).
    - Plans efficient routes using A* pathfinding to systematically explore unknown environments.

_Perception & Target Tracking:_
    - Computer vision with HSV + Hough transforms for colored target detection.
    - Camera-based distance estimation (pinhole model) fused with ultrasonic sonar for robust measurements.

_-Navigation & Control:_
    - Differential drive with four DC motors (L298N drivers, PWM control).
    - Built-in safety features: E-stop, soft-start PWM limits, and motor toggle via UI.
    - Escape behavior using ultrasonic sensors & LIDAR when trapped in confined spaces.

_User Interface & Operations:_
    - Flask-based MJPEG streaming server with live video, state overlays, and occupancy map visualization.
    - Remote control interface with E-stop and motor toggling.
    - Debug-friendly live telemetry for testing and demonstrations.

**Real-world impact:**
This project simulates the fundamentals of autonomous robotics used in:

1) Self-driving cars (LiDAR mapping, path planning, sensor fusion).

2) Mobile exploration robots (frontier exploration for search & rescue, warehouse mapping).

3) Human-robot interaction research (safety features, live observability).

It provides a hands-on robotics testbed for experimenting with AI navigation, perception, and multi-sensor integration. The platform can be extended with reinforcement learning, SLAM algorithms, or additional sensors, making it suitable for both education and research prototyping.

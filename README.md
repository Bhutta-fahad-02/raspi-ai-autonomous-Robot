# raspi-ai-autonomous-Robot


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/3561ecc6-c926-4749-816a-0e860d984c10" />



This project is a miniature robotics platform designed to demonstrate the core capabilities of autonomous vehicles in a low-cost, portable form. Built on a Raspberry Pi with LiDAR, ultrasonic sensors, and camera vision, it integrates mapping, planning, perception, and control into a full-stack robotics system.

# **Capabilities:**

_Mapping & Localization:_ 
    - Builds a real-time occupancy grid from LiDAR scans (Bresenham raytracing).  
    - Maintains robot pose with dead-reckoning motion model.  
    
_Autonomous Exploration:_ 
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

# **Real-world impact:**
This project simulates the fundamentals of autonomous robotics used in:  

1) Self-driving cars (LiDAR mapping, path planning, sensor fusion).  

2) Mobile exploration robots (frontier exploration for search & rescue, warehouse mapping).  

3) Human-robot interaction research (safety features, live observability).  

It provides a hands-on robotics testbed for experimenting with AI navigation, perception, and multi-sensor integration. The platform can be extended with reinforcement learning, SLAM algorithms, or additional sensors, making it suitable for research prototyping.  


Below are images of the completed build:  


# Autonomous Truck: 

<img width="802" height="1078" alt="image" src="https://github.com/user-attachments/assets/effe32e3-a077-429d-ab43-233c7f4f7907" />


<img width="3024" height="4032" alt="image" src="https://github.com/user-attachments/assets/a89fc5b2-adf4-4b22-b14b-ac86fdb18e66" />



<img width="4032" height="3024" alt="image" src="https://github.com/user-attachments/assets/0525ea19-4757-465f-8f96-e1c1c2c2660e" />



# Pi setup: 


<img width="3024" height="4032" alt="image" src="https://github.com/user-attachments/assets/7623abf3-1977-4eac-9292-2400430f89a3" />



# States Diagram: 


<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/9c5e68bd-5c92-4612-803c-e5bddcb611df" />


# UI:

<img width="808" height="760" alt="image" src="https://github.com/user-attachments/assets/ee1179c8-0837-458a-b89c-fac1d04eb35d" />







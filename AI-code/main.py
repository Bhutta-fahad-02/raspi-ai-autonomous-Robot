#!/usr/bin/env python3
"""
Raspberry Pi Robotics Platform — AI Upgrade
- AI Vision: YOLOv8 (fallback to HSV/Hough)
- AI Navigation: Stable-Baselines3 PPO (fallback to A* frontier exploration)
"""

import os, sys, time, math, heapq, threading, json, shutil
import numpy as np
import cv2
import lgpio
from flask import Flask, Response, redirect, url_for, make_response
from rplidar import RPLidar

# ========== CONFIG (EDIT ME) ==========
# --- LiDAR ports ---
DEFAULT_LIDAR_PORT = '/dev/ttyUSB0'
PORT_CANDIDATES = [
    '/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyACM0','/dev/ttyACM1',
    '/dev/cu.usbserial-0001'
]
LIDAR_BAUD = 115200

# --- Map & planning ---
MAP_SIZE_M  = 6.0
RESOLUTION  = 0.05
W = H       = int(MAP_SIZE_M / RESOLUTION)
ORIGIN      = (W//2, H//2)
UNKNOWN, FREE, OCC = -1, 0, 1
NEIGHBORS   = [(1,0),(-1,0),(0,1),(0,-1)]

# --- Ultrasonic pins ---
TRIG1, ECHO1 = 21, 20
TRIG2, ECHO2 = 4,  26
ULTRA_THRESHOLD_CM = 20
STUCK_LIMIT = 5

# --- Motor pins (L298N x2) ---
IN1_A1, IN2_A1, ENA_A1 = 24, 23, 25
IN1_B1, IN2_B1, ENB_B1 = 17, 27, 22
IN1_A2, IN2_A2, ENA_A2 = 6,  12, 5
IN1_B2, IN2_B2, ENB_B2 = 13, 19, 18

PWM_FREQ_HZ = 1000
SAFE_MODE = True          # motors gated by default
DRIVE_DUTY = 60           # % duty for motion (tame for safety)

# --- Motion model (timed open-loop) ---
FORWARD_SPEED = 0.1       # m/s (approx)
TURN_RATE     = 1.0       # rad/s
DT            = 0.1       # s

# --- Classic vision (fallback) ---
HSV_LOW  = np.array([45,100,100])    
HSV_HIGH = np.array([90,255,255])

# --- Target distance overlay ---
SHOW_TARGET_DISTANCE = True
TARGET_DIAMETER_M = 0.06
FOCAL_PIX = 700
CENTER_TOL_PX = 30

# --- Streaming server ---
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 5000
JPEG_QUALITY = 75

# --- AI toggles & model paths ---
USE_CNN_VISION = True
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "training-scripts/target.pt") 
YOLO_FALLBACK = "yolov8n.pt"  

USE_RL_NAV = True
RL_POLICY_PATH = os.environ.get("RL_POLICY_PATH", "training-scripts/ppo_nav.zip")  
RL_ACTIONS = ["FWD", "LEFT", "RIGHT", "REV", "STOP"]  

# --- RL observation config ---
LIDAR_BINS = 36   
LIDAR_CLIP = 2.5     
OBS_INCLUDES_SONAR = True
OBS_INCLUDES_GOAL  = True 

# ========== Camera wrapper ==========
class Camera:
    def __init__(self, width=640, height=480):
        self.use_picam2 = False
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            self.use_picam2 = True
        except Exception:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if self.use_picam2:
            frame = self.picam2.capture_array()
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            return self.cap.read()

    def release(self):
        if self.use_picam2:
            try: self.picam2.stop()
            except Exception: pass
        else:
            try: self.cap.release()
            except Exception: pass

# ========== HTTP stream (MJPEG) ==========
app = Flask(__name__)
_latest_jpeg = None
_jpeg_lock = threading.Lock()

INDEX_HTML = """
<!doctype html>
<title>Pi Car Stream</title>
<style>
 body{font-family:system-ui,Arial;margin:20px;background:#0b0b0b;color:#e7e7e7}
 a.button{display:inline-block;margin:4px 6px;padding:8px 12px;border-radius:10px;background:#2d2d2d;color:#fff;text-decoration:none}
 .pill{padding:6px 10px;background:#1a1a1a;border-radius:999px;margin-left:8px;font-size:12px}
 img{max-width:98vw;border-radius:12px;box-shadow:0 0 20px #000}
</style>
<h1>Pi Car — Live View <span class="pill">motors: {{ motors }}</span> <span class="pill">vision: {{ vision }}</span> <span class="pill">nav: {{ nav }}</span></h1>
<p>
  <a class="button" href="/api/toggle_motors">Toggle Motors</a>
  <a class="button" href="/api/estop">E-Stop</a>
</p>
<img src="/stream.mjpg">
"""

def set_latest_jpeg(frame_bgr):
    global _latest_jpeg
    ok, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if ok:
        with _jpeg_lock:
            _latest_jpeg = buf.tobytes()

@app.route('/')
def index():
    motors = "OFF" if SAFE_MODE else "ON"
    vision = "CNN" if Robot.instance and Robot.instance.using_cnn_vision else "HSV"
    nav    = "RL" if Robot.instance and Robot.instance.using_rl_nav else "A*"
    html = INDEX_HTML.replace("{{ motors }}", motors).replace("{{ vision }}", vision).replace("{{ nav }}", nav)
    return make_response(html)

@app.route('/stream.mjpg')
def stream():
    def gen():
        while True:
            with _jpeg_lock:
                data = _latest_jpeg
            if data is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_motors')
def toggle_motors():
    global SAFE_MODE
    SAFE_MODE = not SAFE_MODE
    return redirect(url_for('index'))

@app.route('/api/estop')
def estop():
    Robot.instance.estop()
    return redirect(url_for('index'))

def start_http_server():
    t = threading.Thread(
        target=lambda: app.run(host=STREAM_HOST, port=STREAM_PORT, debug=False, use_reloader=False),
        daemon=True
    )
    t.start()

# ========== Utilities ==========
def safe_import_ultralytics(weights_hint=None):
    """
    import ultralytics and load YOLO model.
    Returns (model, note_str) or (None, reason)
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        return None, f"ultralytics not available: {e}"


    weights = None
    if weights_hint and os.path.exists(weights_hint):
        weights = weights_hint
    elif os.path.exists(YOLO_FALLBACK):
        weights = YOLO_FALLBACK
    else:
        
        weights = "yolov8n.pt"

    try:
        model = YOLO(weights)
        return model, f"Loaded YOLO weights: {weights}"
    except Exception as e:
        return None, f"Failed to load YOLO weights ({weights}): {e}"

def safe_import_sb3(policy_path):
    """
    import Stable-Baselines3 and load PPO policy.
    Returns (policy, note_str) or (None, reason)
    """
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        return None, f"stable-baselines3 not available: {e}"

    if not os.path.exists(policy_path):
        return None, f"RL policy file not found: {policy_path}"

    try:
        policy = PPO.load(policy_path, device="cpu")
        return policy, f"Loaded PPO policy: {policy_path}"
    except Exception as e:
        return None, f"Failed to load PPO policy: {e}"

def angle_wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

# ========== Robot ==========
class Robot:
    instance = None

    def __init__(self):
        Robot.instance = self

        # GPIO
        self.h = lgpio.gpiochip_open(0)
        # direction pins
        for p in (IN1_A1,IN2_A1,IN1_B1,IN2_B1,IN1_A2,IN2_A2,IN1_B2,IN2_B2):
            lgpio.gpio_claim_output(self.h, p)
        # pwm pins
        for pwm in (ENA_A1,ENB_B1,ENA_A2,ENB_B2):
            lgpio.gpio_claim_output(self.h, pwm)
            lgpio.tx_pwm(self.h, pwm, PWM_FREQ_HZ, 0)
        # ultrasonics
        for trig in (TRIG1, TRIG2):
            lgpio.gpio_claim_output(self.h, trig)
        for echo in (ECHO1, ECHO2):
            lgpio.gpio_claim_input(self.h, echo)

        # LiDAR
        self.lidar = self._open_lidar()
        self.scan_iter = self.lidar.iter_scans()
        self.last_scan = None 

        # Camera
        self.cam = Camera(640, 480)

        # Map/state
        self.grid = np.full((H, W), UNKNOWN, dtype=np.int8)
        self.pose = [0.0, 0.0, 0.0]  
        self.path = []
        self.stuck_counter = 0
        self.state = "Init"

        # AI: vision
        self.using_cnn_vision = False
        self.yolo = None
        if USE_CNN_VISION:
            self.yolo, note = safe_import_ultralytics(YOLO_WEIGHTS)
            if self.yolo:
                self.using_cnn_vision = True
                print("[Vision] CNN enabled:", note)
            else:
                print("[Vision] CNN unavailable, falling back to HSV/Hough:", note)

        # AI: RL navigation
        self.using_rl_nav = False
        self.rl_policy = None
        if USE_RL_NAV:
            self.rl_policy, note = safe_import_sb3(RL_POLICY_PATH)
            if self.rl_policy:
                self.using_rl_nav = True
                print("[Nav] RL policy enabled:", note)
            else:
                print("[Nav] RL unavailable, falling back to A* frontier:", note)

    # ----- hardware utils -----
    def _open_lidar(self):
        for p in PORT_CANDIDATES:
            if os.path.exists(p):
                try:
                    return RPLidar(p, baudrate=LIDAR_BAUD)
                except Exception:
                    pass
        return RPLidar(DEFAULT_LIDAR_PORT, baudrate=LIDAR_BAUD)

    def _pwm_all(self, duty_percent):
        duty = max(0, min(100, int(duty_percent)))
        if SAFE_MODE: duty = 0
        for pwm in (ENA_A1,ENB_B1,ENA_A2,ENB_B2):
            lgpio.tx_pwm(self.h, pwm, PWM_FREQ_HZ, duty)

    def _dir_all(self, fwd=True):
        pairs = ((IN1_A1,IN2_A1),(IN1_B1,IN2_B1),(IN1_A2,IN2_A2),(IN1_B2,IN2_B2))
        if fwd:
            for in1,in2 in pairs: lgpio.gpio_write(self.h, in1,1); lgpio.gpio_write(self.h, in2,0)
        else:
            for in1,in2 in pairs: lgpio.gpio_write(self.h, in1,0); lgpio.gpio_write(self.h, in2,1)

    def _spin(self, ccw=True):
        left  = ((IN1_A1,IN2_A1),(IN1_A2,IN2_A2))
        right = ((IN1_B1,IN2_B1),(IN1_B2,IN2_B2))
        if ccw:
            for in1,in2 in left:  lgpio.gpio_write(self.h, in1,0); lgpio.gpio_write(self.h, in2,1)
            for in1,in2 in right: lgpio.gpio_write(self.h, in1,1); lgpio.gpio_write(self.h, in2,0)
        else:
            for in1,in2 in left:  lgpio.gpio_write(self.h, in1,1); lgpio.gpio_write(self.h, in2,0)
            for in1,in2 in right: lgpio.gpio_write(self.h, in1,0); lgpio.gpio_write(self.h, in2,1)

    def stop_all(self):
        for in1,in2 in ((IN1_A1,IN2_A1),(IN1_B1,IN2_B1),(IN1_A2,IN2_A2),(IN1_B2,IN2_B2)):
            lgpio.gpio_write(self.h, in1,0); lgpio.gpio_write(self.h, in2,0)
        self._pwm_all(0)

    def estop(self):
        global SAFE_MODE
        SAFE_MODE = True
        self.state = "E-STOP"
        self.stop_all()

    # ----- ultrasonics -----
    def _pulse(self, trig, echo, timeout=0.04):
        lgpio.gpio_write(self.h, trig, 0); time.sleep(5e-6)
        lgpio.gpio_write(self.h, trig, 1); time.sleep(1e-5)
        lgpio.gpio_write(self.h, trig, 0)
        t0 = time.time()
        while lgpio.gpio_read(self.h, echo)==0:
            if time.time()-t0>timeout: return None, None
        t1 = time.time()
        while lgpio.gpio_read(self.h, echo)==1:
            if time.time()-t1>timeout: return t1, None
        t2 = time.time()
        return t1, t2

    def get_distance_cm(self, trig, echo):
        t1,t2 = self._pulse(trig, echo)
        if t1 is None or t2 is None: return float('inf')
        return ((t2 - t1) * 34300) / 2.0

    def read_ultrasonics(self):
        return self.get_distance_cm(TRIG1, ECHO1), self.get_distance_cm(TRIG2, ECHO2)

    # ----- camera-based distance from target -----
    def estimate_cam_distance_m(self, radius_px: int):
        if not SHOW_TARGET_DISTANCE or radius_px <= 0:
            return None
        return (FOCAL_PIX * TARGET_DIAMETER_M) / (2.0 * float(radius_px))

    # ----- motion & pose -----
    def execute_motion(self, v, w, dt=DT):
        if   v>0 and w==0: self._dir_all(True);  self._pwm_all(DRIVE_DUTY)
        elif v<0 and w==0: self._dir_all(False); self._pwm_all(DRIVE_DUTY)
        elif w>0 and v==0: self._spin(ccw=True); self._pwm_all(DRIVE_DUTY)
        elif w<0 and v==0: self._spin(ccw=False);self._pwm_all(DRIVE_DUTY)
        else:              self.stop_all()
        time.sleep(dt)
        self.stop_all()

        # pose integration
        x,y,theta = self.pose
        theta2 = theta + w*dt
        x2     = x + v*math.cos(theta)*dt
        y2     = y + v*math.sin(theta)*dt
        self.pose = [x2,y2,theta2]

    # ----- mapping (occupancy grid) -----
    def world_to_cell(self, x, y):
        return int(x/RESOLUTION)+ORIGIN[0], ORIGIN[1]-int(y/RESOLUTION)

    def bres(self, x0, y0, x1, y1):
        dx, sx = abs(x1-x0), 1 if x1>x0 else -1
        dy, sy = abs(y1-y0), 1 if y1>y0 else -1
        err = dx - dy
        while True:
            yield x0, y0
            if x0==x1 and y0==y1: break
            e2 = err*2
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy

    def build_grid(self, scan):
        self.last_scan = scan
        x0,y0,th = self.pose
        c0 = self.world_to_cell(x0,y0)
        for _, ang_deg, dist_mm in scan:
            if dist_mm <= 0: continue
            d = dist_mm/1000.0
            thw = th + math.radians(ang_deg)
            xw,yw = x0 + d*math.cos(thw), y0 + d*math.sin(thw)
            c1 = self.world_to_cell(xw,yw)
            for cx,cy in self.bres(*c0,*c1):
                if 0<=cx<W and 0<=cy<H: self.grid[cy,cx] = FREE
            cx,cy = c1
            if 0<=cx<W and 0<=cy<H: self.grid[cy,cx] = OCC

    # ----- classic planning -----
    def find_frontiers(self):
        out=[]
        for y in range(1,H-1):
            for x in range(1,W-1):
                if self.grid[y,x]==FREE and any(self.grid[y+dy,x+dx]==UNKNOWN for dx,dy in NEIGHBORS):
                    out.append((x,y))
        return out

    def astar(self, start, goal):
        pq=[(abs(start[0]-goal[0])+abs(start[1]-goal[1]),0,start,None)]
        came, gscore = {}, {start:0}
        while pq:
            f,g,n,parent = heapq.heappop(pq)
            if n in came: continue
            came[n]=parent
            if n==goal:
                path=[n]
                while came[path[-1]]: path.append(came[path[-1]])
                return path[::-1]
            for dx,dy in NEIGHBORS:
                nb=(n[0]+dx,n[1]+dy)
                x,y=nb
                if not (0<=x<W and 0<=y<H): continue
                if self.grid[y,x]==OCC: continue
                ng=g+1
                if ng<gscore.get(nb,1e9):
                    gscore[nb]=ng
                    h=abs(nb[0]-goal[0])+abs(nb[1]-goal[1])
                    heapq.heappush(pq,(ng+h,ng,nb,parent if False else n))
        return []

    # ----- CNN vision detect (Fallback: OpenCV) -----
    def detect_target(self, frame_bgr):
        # CNN first
        if self.using_cnn_vision and self.yolo is not None:
            try:
                # inference
                results = self.yolo(frame_bgr, verbose=False)
                if results and len(results[0].boxes) > 0:
                    # pick the largest box / first
                    boxes = results[0].boxes
                    areas = []
                    for i in range(len(boxes)):
                        x1,y1,x2,y2 = map(int, boxes.xyxy[i].tolist())
                        areas.append((i, (x2-x1)*(y2-y1)))
                    idx = max(areas, key=lambda t:t[1])[0]
                    x1,y1,x2,y2 = map(int, boxes.xyxy[idx].tolist())
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    r = int(0.5*max(x2-x1, y2-y1))
                    return (cx, cy, r)
            except Exception as e:
                #drop to fallback
                print("[Vision] YOLO error, fallback to HSV:", e)
                self.using_cnn_vision = False

        # Fallback: HSV/Hough
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=100, param1=50, param2=20,
                                   minRadius=10, maxRadius=200)
        if circles is None: return None
        x,y,r = max(np.uint16(np.around(circles[0])).tolist(), key=lambda c:c[2])
        return (int(x),int(y),int(r))

    def approach_target(self):
        self.state = "Approaching"
        while True:
            ok, frame = self.cam.read()
            if not ok: continue
            det = self.detect_target(frame)
            if not det:
                self.execute_motion(0, TURN_RATE, DT); continue
            x,y,_ = det
            err = x - frame.shape[1]//2
            if abs(err)>20:
                self.execute_motion(0, TURN_RATE*math.copysign(1,err), DT)
            else:
                d1,d2 = self.read_ultrasonics()
                if min(d1,d2) <= 25:
                    self.stop_all(); print("Reached target."); return
                self.execute_motion(FORWARD_SPEED,0,DT)

    # ----- RL helper: observation builder -----
    def _lidar_bins(self):
        """
        Convert last LiDAR scan to fixed-size bins of min distance (meters).
        """
        bins = np.full(LIDAR_BINS, LIDAR_CLIP, dtype=np.float32)
        if self.last_scan is None: return bins
        # scan entries: (quality, angle_deg, dist_mm)
        for _, ang_deg, dist_mm in self.last_scan:
            if dist_mm <= 0: continue
            d_m = min(dist_mm/1000.0, LIDAR_CLIP)
            # map angle to [0, 360)
            a = (ang_deg % 360.0)
            bi = int((a / 360.0) * LIDAR_BINS)
            bins[bi] = min(bins[bi], d_m)
        return bins

    def _nearest_frontier_and_heading(self):
        """
        Find nearest frontier cell and compute goal vector & heading error.
        Returns (dx, dy, heading_err) in world coordinates (meters, radians).
        """
        start = self.world_to_cell(*self.pose[:2])
        fronts = self.find_frontiers()
        if not fronts:
            return 0.0, 0.0, 0.0
        goal = min(fronts, key=lambda c:(c[0]-start[0])**2+(c[1]-start[1])**2)
        gx = (goal[0]-ORIGIN[0]) * RESOLUTION
        gy = (ORIGIN[1]-goal[1]) * RESOLUTION
        dx, dy = gx - self.pose[0], gy - self.pose[1]
        desired = math.atan2(dy, dx)
        herr = angle_wrap(desired - self.pose[2])
        return dx, dy, herr

    def build_observation(self):
        parts = [self._lidar_bins()]
        if OBS_INCLUDES_SONAR:
            d1,d2 = self.read_ultrasonics()
            s = np.array([min(d1, 500.0)/100.0, min(d2, 500.0)/100.0], dtype=np.float32)  # ~meters
            parts.append(s)
        if OBS_INCLUDES_GOAL:
            dx, dy, herr = self._nearest_frontier_and_heading()
            parts.append(np.array([dx, dy, herr], dtype=np.float32))
        obs = np.concatenate(parts, axis=0).astype(np.float32)
        return obs

    def rl_decide_and_act(self):
        """
        Use RL policy to choose a discrete action and execute it for DT.
        """
        obs = self.build_observation()
        try:
            action, _ = self.rl_policy.predict(obs, deterministic=True)
        except Exception as e:
            print("[RL] Predict failed, fallback to A* this loop:", e)
            return False  # signal fallback

        act = RL_ACTIONS[int(action) % len(RL_ACTIONS)]
        if act == "FWD":
            self.execute_motion(FORWARD_SPEED, 0, DT)
        elif act == "LEFT":
            self.execute_motion(0, TURN_RATE, DT)
        elif act == "RIGHT":
            self.execute_motion(0, -TURN_RATE, DT)
        elif act == "REV":
            self.execute_motion(-FORWARD_SPEED, 0, DT)
        else:
            self.stop_all()
            time.sleep(DT)
        return True

    # ----- main -----
    def run(self):
        start_http_server()
        print(f"HTTP stream at http://{STREAM_HOST}:{STREAM_PORT}/ (SSH port-forward if remote)")
        gui = bool(os.environ.get('DISPLAY','')) or sys.platform in ('win32','darwin')

        try:
            while True:
                # 1) Vision:  target approach demo
                ok, cam_frame = self.cam.read()
                if not ok: continue
                det = self.detect_target(cam_frame)
                if det:
                    self.approach_target()
                    det = None  # avoid stale overlay

                # 2) LiDAR update + map
                try:
                    scan = next(self.scan_iter)
                except Exception as e:
                    print("LiDAR error:", e)
                    continue
                self.build_grid(scan)

                # 3) Near-field reactive avoid
                d1,d2 = self.read_ultrasonics()
                if d1<ULTRA_THRESHOLD_CM and d2<ULTRA_THRESHOLD_CM:
                    self.stuck_counter += 1
                    self.state = "Avoiding"
                else:
                    self.stuck_counter = 0
                if self.stuck_counter > STUCK_LIMIT:
                    self.execute_motion(-FORWARD_SPEED,0,DT)
                    self.execute_motion(0,TURN_RATE,DT)
                    self.stuck_counter = 0
                    
                    continue

                # 4) Navigation: RL if available, else A*
                acted = False
                if self.using_rl_nav and self.rl_policy is not None:
                    self.state = "RL-NAV"
                    acted = self.rl_decide_and_act()

                if not acted:
                    # Classical A* frontier exploration
                    if not self.path:
                        self.state = "Planning"
                        start = self.world_to_cell(*self.pose[:2])
                        fronts = self.find_frontiers()
                        if fronts:
                            goal = min(fronts, key=lambda c:(c[0]-start[0])**2+(c[1]-start[1])**2)
                            self.path = self.astar(start, goal)

                    if self.path:
                        self.state = "Exploring"
                        nx,ny = self.path[0]
                        wx = (nx-ORIGIN[0]) * RESOLUTION
                        wy = (ORIGIN[1]-ny) * RESOLUTION
                        dx,dy = wx-self.pose[0], wy-self.pose[1]
                        dist  = math.hypot(dx,dy)
                        desired = math.atan2(dy,dx)
                        err = angle_wrap(desired - self.pose[2])
                        if abs(err)>0.2:
                            self.execute_motion(0, TURN_RATE*math.copysign(1,err), DT)
                        elif dist>0.1:
                            self.execute_motion(FORWARD_SPEED,0,DT)
                        else:
                            self.path.pop(0)
                    else:
                        self.state = "Idle"
                        self.stop_all()

                # 5) Overlays & stream
                disp = cam_frame.copy()
                target_dist_text = None

                if det:
                    x,y,r = det
                    cv2.circle(disp,(x,y),r,(0,255,0),2)
                    cv2.putText(disp,"TARGET",(x-20,y-20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                    z_m = self.estimate_cam_distance_m(r)
                    if z_m is not None and math.isfinite(z_m):
                        target_dist_text = f"{z_m:.2f} m (cam)"
                    if abs(x - disp.shape[1]//2) < CENTER_TOL_PX:
                        sonar_cm = min(d1, d2)
                        if math.isfinite(sonar_cm):
                            if target_dist_text:
                                target_dist_text += f" | {sonar_cm:.0f} cm (sonar)"
                            else:
                                target_dist_text = f"{sonar_cm:.0f} cm (sonar)"

                if SHOW_TARGET_DISTANCE and target_dist_text:
                    cv2.putText(disp, f"TargetDist: {target_dist_text}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                cv2.putText(
                    disp,
                    f"State: {self.state} | Motors:{'OFF' if SAFE_MODE else 'ON'} | Vision:{'CNN' if self.using_cnn_vision else 'HSV'} | Nav:{'RL' if self.using_rl_nav else 'A*'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2
                )

                # Mini-map
                map_img = np.zeros((H, W, 3), dtype=np.uint8)
                map_img[self.grid==UNKNOWN] = (128,128,128)
                map_img[self.grid==FREE]    = (255,255,255)
                map_img[self.grid==OCC]     = (0,0,0)
                for cell in self.path:
                    cv2.circle(map_img, cell, 1, (255,0,0), -1)
                px,py = self.world_to_cell(*self.pose[:2])
                if 0<=px<W and 0<=py<H:
                    cv2.circle(map_img,(px,py),3,(0,0,255),-1)
                map_small = cv2.resize(map_img,(disp.shape[1]//3, disp.shape[0]//3),
                                       interpolation=cv2.INTER_NEAREST)
                mh,mw = map_small.shape[:2]
                disp[0:mh, disp.shape[1]-mw:disp.shape[1]] = map_small

                set_latest_jpeg(disp)

                if gui:
                    cv2.imshow("Pi Car", disp)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'): break
                    if k == ord('m'):
                        global SAFE_MODE
                        SAFE_MODE = not SAFE_MODE
                    if k == ord('e'):
                        self.estop()
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()
            try: self.lidar.stop(); self.lidar.disconnect()
            except Exception: pass
            self.cam.release()
            cv2.destroyAllWindows()
            lgpio.gpiochip_close(self.h)

# ======== entry ========
if __name__ == '__main__':
    Robot().run()

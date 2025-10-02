
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RandomWorldNavEnv(gym.Env):
    def __init__(self, max_dist=2.5, lidar_bins=36, max_steps=300, visualize=False):
        super().__init__()
        self.max_dist = max_dist
        self.lidar_bins = lidar_bins
        self.max_steps = max_steps
        self.visualize = visualize

        self.observation_space = gym.spaces.Box(
            low=0, high=max_dist, shape=(lidar_bins+2+3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)

        self.fig = None
        self.reset()

    def reset(self):
        self.steps = 0
        # random world size: 3x3, 6x6, or 12x12
        self.world_size = np.random.choice([3.0, 6.0, 12.0])
        # Random walls
        self.walls = []
        n_walls = np.random.randint(2, 5)
        for _ in range(n_walls):
            x1, y1 = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
            w, h = np.random.uniform(0.5, self.world_size/3), np.random.uniform(0.5, self.world_size/3)
            self.walls.append(((x1,y1),(x1+w,y1+h)))
        # Random start + heading
        self.pos = np.random.uniform(-self.world_size/2, self.world_size/2, size=2)
        self.heading = np.random.uniform(-np.pi, np.pi)
        # Random goal
        self.goal = np.random.uniform(-self.world_size/2, self.world_size/2, size=2)
        return self._obs()

    def _cast_ray(self, angle):
        x0,y0 = self.pos
        dx,dy = np.cos(angle), np.sin(angle)
        for d in np.linspace(0,self.max_dist,50):
            x,y = x0+d*dx, y0+d*dy
            # Out of bounds
            if abs(x) > self.world_size/2 or abs(y) > self.world_size/2:
                return d
            # Hit wall
            for (x1,y1),(x2,y2) in self.walls:
                if x1<=x<=x2 and y1<=y<=y2:
                    return d
        return self.max_dist

    def _lidar_scan(self):
        return np.array([
            self._cast_ray(self.heading + 2*np.pi*i/self.lidar_bins)
            for i in range(self.lidar_bins)
        ],dtype=np.float32)

    def _obs(self):
        lidar = self._lidar_scan()
        sonar_left = self._cast_ray(self.heading+np.radians(30))
        sonar_right= self._cast_ray(self.heading-np.radians(30))
        dx,dy = self.goal - self.pos
        heading_err = np.arctan2(dy,dx) - self.heading
        return np.concatenate([lidar,[sonar_left,sonar_right],[dx,dy,heading_err]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        reward, done = -0.01, False

        # Move
        if action==0: self.pos += [np.cos(self.heading)*0.1, np.sin(self.heading)*0.1]
        elif action==1: self.heading += 0.2
        elif action==2: self.heading -= 0.2
        elif action==3: self.pos -= [np.cos(self.heading)*0.1, np.sin(self.heading)*0.1]

        # Collisions
        for (x1,y1),(x2,y2) in self.walls:
            if x1<=self.pos[0]<=x2 and y1<=self.pos[1]<=y2:
                reward -= 1; done=True

        # Goal
        if np.linalg.norm(self.goal-self.pos)<0.3:
            reward += 10; done=True

        if self.steps>=self.max_steps:
            done=True

        if self.visualize: self.render()

        return self._obs(), reward, done, {}

    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        self.ax.clear()
        self.ax.set_xlim(-self.world_size/2, self.world_size/2)
        self.ax.set_ylim(-self.world_size/2, self.world_size/2)
        # Draw walls
        for (x1,y1),(x2,y2) in self.walls:
            self.ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, color="black"))
        # Draw goal
        self.ax.plot(self.goal[0], self.goal[1], "go", markersize=10)
        # Draw robot
        self.ax.plot(self.pos[0], self.pos[1], "ro")
        # Draw heading
        self.ax.arrow(self.pos[0],self.pos[1],
                      0.3*np.cos(self.heading),0.3*np.sin(self.heading),
                      head_width=0.1,color="r")
        # Draw lidar rays
        for i in range(self.lidar_bins):
            angle = self.heading + 2*np.pi*i/self.lidar_bins
            d = self._cast_ray(angle)
            x,y = self.pos[0]+d*np.cos(angle), self.pos[1]+d*np.sin(angle)
            self.ax.plot([self.pos[0],x],[self.pos[1],y],"b-",alpha=0.3)
        plt.pause(0.001)

# ====== Train PPO ======
if __name__=="__main__":
    env = make_vec_env(lambda: RandomWorldNavEnv(visualize=False), n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_nav_tb/")
    model.learn(total_timesteps=300_000)
    model.save("ppo_nav.zip")
    print("Saved PPO policy: ppo_nav.zip")

    # Test visualization with 1 env
    test_env = RandomWorldNavEnv(visualize=True)
    obs = test_env.reset()
    for _ in range(200):
        action = test_env.action_space.sample()
        obs,r,done,_ = test_env.step(action)
        if done: obs = test_env.reset()

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.config import Config
from PIL import Image as PILImage
from PIL import ImageDraw , ImageOps
import  numpy as np
import os
import math
import time
from random import randint
import TD3 as RL
import torch
import gym 
import  inspect


# Adding this line if we don't want the right click to put a red point
# Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# observation_space = gym.spaces.Box(low=0, high=255, shape=(40, 40, 1)) # 40 * 40 
action_space = gym.spaces.box.Box(low = np.array([-5]), high = np.array([5]), dtype=np.float32) # Rotation

seed = 0 # Random seed number
start_timesteps = 10 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

torch.manual_seed(seed)
np.random.seed(seed)

# state_dim = observation_space.shape[0]
action_dim = action_space.shape[0]
max_action = float(action_space.high[0])

# AI
policy = RL.TD3(action_dim,max_action)

replay_buffer =  RL.ReplayBuffer()

#  Initializing map
first_update = True
def init():

    print ("Entered init method")

    global goal_x
    global goal_y
    global swap
    global distance
    global first_update
    global timesteps_since_eval
    global episode_num
    global done
    global t0
    global  total_timesteps
    global i

    i = 0
    goal_x = 1415
    goal_y = 622
    swap = 0
    first_update = False
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
 
class TestCar(Widget):

    angle = NumericProperty(0.)
    rotation = NumericProperty(0.)
    velocity_x = NumericProperty(0.) 
    velocity_y = NumericProperty(0.)
    deceleration_x = NumericProperty(0.)
    deceleration_y = NumericProperty(0.)

    velocity = ReferenceListProperty(velocity_x, velocity_y)
    deceleration = ReferenceListProperty(deceleration_x, deceleration_y)

    def rotate(self,rotation):
        print ("Entered rotate method")
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = float(rotation)
        self.angle = self.angle + self.rotation
    


    # def move(self):
    #     self.pos =Vector(*self.velocity) + self.pos
    #     print ("pos3 :",self.pos , "  vel3 :  " , self.velocity)


class TestGame(Widget):

    testcar = ObjectProperty(None)

    def serve_car(self):
        print ("Entered  method: ", inspect.stack()[0][3])

        self.testcar.center = self.center
        self.testcar.velocity = Vector(1, 0)
        self.testcar.deceleration = Vector(0,0)
        self.testcar.dest_reward = False
        self.testcar.wall_reward = False

    def apply_action(self,action):
        print ("Entered  method: ", inspect.stack()[0][3])
        # self.testcar.accelerate(action[0]) # accelerate 
        self.testcar.rotate(action[0]) # rotation 
        # self.testcar.decelerate(action[2]) # brake 
        # self.testcar.move() # no_op
    
    def get_screen(self,center_x,center_y,angle):
        
        print ("Entered  method: ", inspect.stack()[0][3])
        #rotates point `A` about point `B` by `angle` radians clockwise.
        def rotated_about(ax, ay, bx, by, angle):
            radius = math.sqrt((by - ay)**2 + (bx - ax)**2)
            angle += math.atan2(ay-by, ax-bx)
            return (
                round(bx + radius * math.cos(angle)),
                round(by + radius * math.sin(angle))
            ) 

        car_center = [center_x , center_y]
        car_angle = - angle
        car_length, car_width = 16 , 25

        triangle_vertices = (
            # (car_center[0] + car_width / 2, car_center[1] + car_length / 2),
            (car_center[0] + car_width , car_center[1] ),
            (car_center[0] - car_width / 2, car_center[1] - car_length / 2),
            (car_center[0] - car_width / 2, car_center[1] + car_length / 2)
        )

        triangle_vertices = [rotated_about(x,y, car_center[0], car_center[1], math.radians(car_angle)) for x,y in triangle_vertices]

        path=os.path.normpath("./images/MASK1.png")

        img = PILImage.open(path)
        draw = ImageDraw.Draw(img)

        draw.polygon(triangle_vertices, fill=(0,255,0))
        (width, height) = (img.width // 3, img.height // 3)
        im_resized_image = img.resize((width , height))

        car_center[0] = car_center[0]//3 
        car_center[1] = car_center[1]//3 

        pad_im_resized_image = ImageOps.expand(im_resized_image, border = 20 , fill =(255,255,255) )   

        car_center[0] = car_center[0] + 20 
        car_center[1] = car_center[1] + 20

        # Co-ordinate points for cropping (40 x 40)
        left ,top , right ,bottom = (car_center[0]//3 - 20) , (car_center[1]//3 - 20 ) , (car_center[0]//3 + 20) , (car_center[1]//3 + 20 )

        im1 = im_resized_image.crop((left, top, right, bottom))
        im2 = ImageOps.grayscale(im1)  

        # Cropped image of 40 x 40  in array format .
        np_array_cutout = np.asarray(im2)/255
        return np_array_cutout

    def _get_state(self):
        print ("Entered  method: ", inspect.stack()[0][3])
        return self.get_screen(self.testcar.x , self.testcar.x , self.testcar.angle)

    def reset(self):
        print ("Entered  method: ", inspect.stack()[0][3])
        self.testcar.x = randint (0,longueur)
        self.testcar.y = randint (0,largeur)
        self.testcar.angle = randint(0,90)
        return self._get_state()

    def step(self,action):
        global done
        print ("Entered  method: ", inspect.stack()[0][3])
        if action is not None:
            self.apply_action(action)

        new_obs = self._get_state()

        Living_Penalty = -1

        dest_bool = self.testcar.dest_reward
        wall_bool = self.testcar.wall_reward

        if dest_bool :
            Destination_reward = 500
        else :
            Destination_reward = 0

        if wall_bool:
            Wall_reward = - 10 
        else :  
            Wall_reward =0 

        if dest_bool or wall_bool :
            done = True
        else:
            done = False

        reward = Living_Penalty + Destination_reward + Wall_reward 

        return new_obs , reward , done 

    # def evaluate_policy(policy, eval_episodes=10):
    #     avg_reward = 0.
    #     for _ in range(eval_episodes):
    #         env.render
    #         obs = env.reset()
    #         done = False
    #         while not done:
    #         action = policy.select_action(np.array(obs))
    #         obs, reward, done, _ = env.step(action)
    #         avg_reward += reward
    #     avg_reward /= eval_episodes
    #     print ("---------------------------------------")
    #     print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    #     print ("---------------------------------------")
    #     return avg_reward


    def update(self, dt):

        print ("Entered  method: ", inspect.stack()[0][3])

        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global first_update
        global timesteps_since_eval
        global episode_num
        global done
        global t0
        global  total_timesteps
        global i
        
        longueur = self.width
        largeur = self.height
        max_timesteps = 500000

        if first_update:
            init()

        # Training
        # We start the main loop over 500,000 timesteps
        while total_timesteps < max_timesteps:
            print ("total_timesteps:" ,total_timesteps )
            # If the episode is done
            if done:
            
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

                # Add evaluation 

                # When the training step is done, we reset the state of the environment
                obs = self.reset()

                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_reward = 0 
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = action_space.sample()
                print ("sample_action_",i," : ",action)
                i+=1
            else: # After 10000 timesteps, we switch to the model
                action = policy.select_action(obs)
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, 
                size=action_space.shape[0])).clip(action_space.low, action_space.high)
  
            new_obs, reward, done = self.step(action)
            
            # We check if the episode is done
            done_bool = 1.0 if episode_timesteps + 1 == 5000 else float(done)
            
            # We increase the total reward
            episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        distance = np.sqrt((self.testcar.x - goal_x)**2 + (self.testcar.y - goal_y)**2)

        if self.testcar.x < 20:
            self.testcar.x = 20
            self.testcar.wall_reward

        if self.testcar.x > self.width - 20:
            self.testcar.x = self.width - 20
            self.testcar.wall_reward

        if self.testcar.y < 20:
            self.testcar.y = 20
            self.testcar.wall_reward

        if self.testcar.y > self.height - 20:
            self.testcar.y = self.height - 20
            self.testcar.wall_reward

        if distance < 25:
            self.testcar.dest_reward = True

            if swap == 1:
                goal_x = 1415
                goal_y = 622
                swap = 0
            
            elif swap == 2 :
                goal_x = 495
                goal_y = 660 - 50
                swap = 1

            elif swap == 3 :
                goal_x = 820
                goal_y = 660 - 630
                swap = 2
            
            elif swap == 4 :
                goal_x = 40
                goal_y = 660 - 570
                swap = 3

            else:
                goal_x = 585
                goal_y = 660 - 300
                swap = 4
        # last_distance = distance
        
class TestApp(App):
    def build(self):
        self.parent = TestGame()
        self.parent.serve_car()
        Clock.schedule_interval(self.parent.update, 1.0/60.0)
        return self.parent 

if __name__ == '__main__':
    TestApp().run()
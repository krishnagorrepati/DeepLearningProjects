from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty,BoundedNumericProperty
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
import TD3_1 as RL
import torch
import gym 
import  inspect   
import pickle
import matplotlib.pyplot as plt


# Adding this line if we don't want the right click to put a red point
# Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# observation_space = gym.spaces.Box(low=0, high=255, shape=(40, 40, 1)) # 40 * 40
action_space = gym.spaces.box.Box(low = np.array([-10,-0.5]), high = np.array([10,2]), dtype=np.float32) # Rotation,Acceleration , Braking (Deceleration) 
# action_space = gym.spaces.box.Box(low = np.array([-30]), high = np.array([30]), dtype=np.float32) # Rotation,Acceleration , Braking (Deceleration)

seed = 142 # Random seed number
start_timesteps = 10000# Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5000 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.2 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

file_name = "%s_%s_%s" % ("TD3", "car_env", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

torch.manual_seed(seed)
np.random.seed(seed)

# state_dim = observation_space.shape[0]
state_dim = 6 # pos_x,pos_y,velocity_x,velocity_y,orientation,-orientation
action_dim = action_space.shape[0]
max_action = action_space.high
min_action = action_space.low

# AI
policy = RL.TD3(state_dim,action_dim,max_action,min_action)
# policy.load(file_name, 'pytorch_models/')


replay_buffer =  RL.ReplayBuffer()

#  Initializing map
first_update = True
def init():

    # print ("Entered init method")

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
    global last_distance
    global episode_timesteps


    i =0 
    last_distance = 0
    goal_x = 760
    goal_y = 350
    swap = 0
    first_update = False
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    t0 = time.time()

 
class TestCar(Widget):

    angle = BoundedNumericProperty(0.)
    rotation = BoundedNumericProperty(0.)
    velocity_x = BoundedNumericProperty(0.) 
    velocity_y = BoundedNumericProperty(0.)
    deceleration_x = BoundedNumericProperty(0.)
    deceleration_y = BoundedNumericProperty(0.)
    acceleration_x = BoundedNumericProperty(0.)
    acceleration_y = BoundedNumericProperty(0.)

    velocity = ReferenceListProperty(velocity_x, velocity_y)
    deceleration = ReferenceListProperty(deceleration_x, deceleration_y)
    acceleration = ReferenceListProperty(acceleration_x, acceleration_y)


    def rotate(self,rotation):
        # print ("Entered rotate method")
        self.velocity = Vector(*self.velocity).rotate(self.angle)
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

    def accelerate(self,acceleration_x):
        # print ("Entered  method: ", inspect.stack()[0][3])
        acceleration_x = float(acceleration_x)
        self.velocity = Vector(*self.velocity) + Vector(acceleration_x,0)
        if self.velocity_x  > 2 : # Speed Governer
            self.velocity_x = 2
        elif(self.velocity_x < -1) : 
            self.velocity_x = -1
        if(self.velocity_y > 1) : # To avoid car drifting moving upwards
            self.velocity_y =1
        elif(self.velocity_y < -0.5): # To avoid car drifting downwards
            self.velocity_y = -0.5
        self.pos = Vector(*self.velocity) + self.pos # change property type
        # print ("pos1 :",self.pos , "  vel1 :  " , self.velocity)

    # def decelerate(self,deceleration_x):
    #     # print ("Entered  method: ", inspect.stack()[0][3])
    #     deceleration_x = float(deceleration_x)
    #     self.velocity = Vector(*self.velocity) - Vector(deceleration_x,0)
    #     if self.velocity_x > 2 :
    #         self.velocity_x = 2
    #     elif(self.velocity_x < 0) : # To avoid car moving backwards
    #         self.velocity_x = 0
    #     if(self.velocity_y < -1): # To avoid car drifting dowanwards
    #         self.velocity_y = -1
    #     self.pos = Vector(*self.velocity) + self.pos
        # print ("pos2 :",self.pos , "  vel2 :  " , self.velocity)

    # def move(self):
    #     self.pos =Vector(*self.velocity) + self.pos
    #     print ("pos3 :",self.pos , "  vel3 :  " , self.velocity)


class TestGame(Widget):

    testcar = ObjectProperty(None)

    def serve_car(self):
        # print ("Entered  method: ", inspect.stack()[0][3])

        self.testcar.center = self.center
        self.testcar.velocity = Vector(2, 0)
        self.testcar.deceleration = Vector(0,0)
        self.testcar.acceleration = Vector(0,0)
        self.testcar.dest_reward = False
        self.testcar.wall_reward = False
        

    def apply_action(self,action):
        # print ("Entered  method: ", inspect.stack()[0][3])
        self.testcar.rotate(action[0]) # rotation 
        self.testcar.accelerate(action[1]) # accelerate 
        # self.testcar.decelerate(action[2]) # brake 
        # self.testcar.move() # no_op
    
    def get_screen(self,center_x,center_y,angle):
        
        # print ("Entered  method: ", inspect.stack()[0][3])
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
        # print(triangle_vertices)
        path=os.path.normpath("C:\\Users\\vegorrep\\Documents\\EVA_Assignments\\PH2_A10_TD3\\Session7Small\\images\\MASK1.png")

        img = PILImage.open(path)
        draw = ImageDraw.Draw(img)

        draw.polygon(triangle_vertices, fill=(255,0,0))
        (width, height) = (img.width // 3, img.height // 3)
        # img.save("original.png")
        # print("original img size : ",img.size)
        # print ("original centers: " , car_center)

        im_resized_image = img.resize((width , height))
        # im_resized_image.save("resized.png")
        # print ("resized_img size : ",im_resized_image.size)
        # print ("resized_img centers :" ,car_center[0]//3 , car_center[1]//3 )

        car_center[0] = car_center[0]//3 
        car_center[1] = car_center[1]//3 

        pad_im_resized_image = ImageOps.expand(im_resized_image, border = 20 , fill =(255,255,255) )   
        # pad_im_resized_image.save("padded.png")
        # print ("padded_resized_img size : ",im_resized_image.size)
        # print ("padded_resized_img centers :" ,car_center[0] + 20 , car_center[1] + 20)

        car_center[0] = car_center[0] + 20 
        car_center[1] = car_center[1] + 20

        # Co-ordinate points for cropping (40 x 40)
        
        left ,top , right ,bottom = (car_center[0] - 20) , (car_center[1] - 20 ) , (car_center[0] + 20) , (car_center[1] + 20 )
        # print (left ,top , right ,bottom)
        im1 = pad_im_resized_image.crop((left, top, right, bottom))
        im2 = ImageOps.grayscale(im1) 
        # file_name = "crop_"+str(i)+".png"
        # if i< 10000:
        #  im2.save(file_name) 

        # Cropped image of 40 x 40  in array format .
        np_array_cutout = np.asarray(im2)/255
        return np_array_cutout

    def _get_state(self):
        # print ("Entered  method: ", inspect.stack()[0][3])
        xx = goal_x - self.testcar.x
        yy = goal_y - self.testcar.y
        orientation = Vector(*self.testcar.velocity).angle((xx,yy))/180.
        Car_states = np.array([self.testcar.x , self.testcar.y,self.testcar.velocity_x ,self.testcar.velocity_y,orientation,-orientation])

        return np.array([self.get_screen(self.testcar.x , self.testcar.y , self.testcar.angle) ,Car_states]) 

    def reset(self):

        # print ("Entered  method: ", inspect.stack()[0][3])
        self.testcar.x = randint (0,longueur)
        self.testcar.y = randint (0,largeur)
        self.testcar.angle = 0
        self.testcar.velocity = Vector(2, 0)
        self.testcar.dest_reward = False
        self.testcar.wall_reward = False

        return self._get_state() 
        
    def step(self,action):
        global done
        global distance
        global last_distance
        global episode_timesteps

          
        # print ("Entered  method: ", inspect.stack()[0][3])
        if action is not None:
            self.apply_action(action)

        new_obs = self._get_state()

        Living_Penalty = -4

        dest_bool = self.testcar.dest_reward
        wall_bool = self.testcar.wall_reward

        if dest_bool :
            Destination_reward = 50000
            self.testcar.dest_reward =  False
            print("Episode ended reached destination")
        else :
            Destination_reward = 0


        if wall_bool:
            Wall_reward = -500
            self.testcar.wall_reward = False
            print("Episode ended reached Wall")
        else :  
            Wall_reward =0 

        distance = np.sqrt((self.testcar.x - goal_x)**2 + (self.testcar.y - goal_y)**2)

        if last_distance > distance :
            Distance_reward = 2
        else : 
            Distance_reward = -4
        

        if dest_bool or wall_bool :
            done = True
        else:
            done = False

        # print("Episode timestep : {} Distance_reward {} Destination_reward : {} Wall_reward : {}".format(episode_timesteps,Distance_reward,Destination_reward,Wall_reward))

        reward = Living_Penalty + Destination_reward + Wall_reward + Distance_reward

        return new_obs , reward , done 

    def evaluate_policy(self,policy, eval_episodes=50):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset()
            done = False
            if not done :
                action = policy.select_action(obs)
                obs, reward, done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

    def close(self):
        pickle_out = open("replay_buffer.pickle","wb")
        pickle.dump(replay_buffer, pickle_out)
        pickle_out.close()


    def update(self, dt):

        # print ("Entered  method: ", inspect.stack()[0][3])
        global last_distance
        global distance
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
        global i
        global  total_timesteps
        global episode_timesteps
        global episode_reward
        global obs
        global max_timesteps
        global evaluations
        

        longueur = self.width
        largeur = self.height
        

        print ("--------------------START-------------------")
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)

        if first_update:
            init()
            # obs = self.reset()
            evaluations = [self.evaluate_policy(policy=policy)]
            done = True
            episode_reward = 0 
            max_timesteps = 500000



        # Training
        # We start the main loop over 500,000 timesteps
        # while total_timesteps < max_timesteps:
        print ("total_timesteps:" ,total_timesteps )
        # If the episode is done
        if total_timesteps < max_timesteps :

            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {}  Episode timestep: {} Reward: {}".format(total_timesteps, episode_num,episode_timesteps, episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq,episode_num)

                # Add evaluation 
                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(self.evaluate_policy(policy))              
                    policy.save(file_name, directory="pytorch_models")
                    np.save("results/%s" % (file_name), evaluations)

                # When the training step is done, we reset the state of the environment
                obs = self.reset()
                print ("env reset")

                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_reward = 0 
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if  total_timesteps < start_timesteps:
                action = action_space.sample()
                print( "Pos_x : ",int(self.testcar.x),"Pos_y : ",int(self.testcar.y))
                print ("sample_action_",i," : ",action)

            else: # After 10000 timesteps, we switch to the model
                action = policy.select_action(obs)
                print( "Pos_x : ",int(self.testcar.x),"Pos_y : ",int(self.testcar.y))
                print ("Policy_action_",i-start_timesteps," : ",action)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, 
                    size=action_space.shape[0])).clip(action_space.low, action_space.high)

            new_obs, reward, done = self.step(action)
            
            # We check if the episode is done
            done_bool = 0.0 if episode_timesteps + 1 == 2500 else float(done)
            
            # We increase the total reward
            episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        else:
            evaluations.append(self.evaluate_policy(policy))
            if save_models: policy.save("%s" % (file_name), directory="pytorch_models")
            np.save("results/%s" % (file_name), evaluations)
        
        print("Velocity : ", self.testcar.velocity)
        print("Last_distance : ", last_distance , " Distance : ", distance)
        print("Destination : ", goal_x,goal_y)

        if self.testcar.x < 20:
            self.testcar.x = 20
            self.testcar.wall_reward = True

        if self.testcar.x > self.width - 20:
            self.testcar.x = self.width - 20
            self.testcar.wall_reward = True

        if self.testcar.y < 20:
            self.testcar.y = 20
            self.testcar.wall_reward = True

        if self.testcar.y > self.height - 20:
            self.testcar.y = self.height - 20
            self.testcar.wall_reward =True

        if distance < 25:  
            self.testcar.dest_reward = True

            if swap == 1:
                goal_x = 760
                goal_y = 350
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
        last_distance = distance
        i+=1
        
        
class TestApp(App):
    def build(self):
        self.parent = TestGame()
        self.parent.serve_car()
        Clock.schedule_interval(self.parent.update, 1/60.0)
        return self.parent 

    def on_stop(self):
        print("Closing the window")
        self.parent.close()

if __name__ == '__main__':
    TestApp().run()
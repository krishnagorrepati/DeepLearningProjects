## Reinforcement Learning (Car Game) Project using TD3 algorithm

Steps Followed:-

- Defined models for Actor and Critics and applied to TD3 Algorithm. Integrated the T3D training algorithm in TestGame.Update() function

**Car States**:- Total Car states are six. 

- x position (idle)
- Y position (idle)
- Velocity_x (in x direction), 
- Velocity_y ( in Y direction)
- Positive Orientation 
- Negative Orientation

**Actions**:- There are  two actions

- Rotation
- Acceleration

**Max Actions**:- (-10, +10) Rotation,  (-0.5,2) Acceleration

​      

**test2.py file has the following details**

> - **TestGame class has the below functions**
>   - **apply_action()**
>   - **get_screen()**
>   - **_get_state()  invokes to get_screen() to get patch of image in given (x,y) size.**
>   - **reset()   resets the environment**
>   - **step()  takes action as parameter and invokes apply_action() and calculates next state (observation) by invoking _get_state(). It returns [new_obs , reward , done]**
>   - **update() fills the replay buffer. Once replay buffer is filled for 10000 random actions, it starts training the car using TD3 model.**
>
> - **TestApp class creates instance of TestGame() and runs the TestGame.serve_car()**
>
> - **TestApp().run() will start running the Car Game.**
>
> 



**test.kv file has following objects to be implemented in test_latest.py file**

> - **TestCar**
>
> - **TestGame**
>
>   

**TD3_1.py has the below obects:-**

> **ReplayBuffer(object): has below functions**
>
> - **add()  adds the transition [state, next_state, action, reward, done] to its internal storage**
> - **_init__()**
> - **sample() returns the batch of transitions**
>
> **TD3() class where TD3 algorithm is implemented.**



#### Actor and Critic Models are defined and T3D algorithm is implemented

**Actor Model**

```
1).actor_2d takes image cropped from the map in 2 dimensional format
2).actor_1d takes actions in 1 dimensional format which gives the position of the car in the map.
 
 self.actor_2d = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 64 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(4)
        # nn.BatchNorm1d(64)                    # check if needed
    )

    self.actor_1d = nn.Sequential(
        nn.Linear(state_dim,50),
        # nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Linear(50,100)
        # nn.BatchNorm1d(100)                 # check if needed
    )

    self.actor_merged = nn.Sequential(
        nn.Linear(100+64 , 200), # Combining feature vectors
        # nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200,250),
        # nn.BatchNorm1d(250),
        nn.ReLU(),
        nn.Linear(250,75),
        nn.ReLU(),
        nn.Linear(75,action_dim)
    )

    self.max_action = max_action
    
def forward(self, state_img,state_pos):

    x = self.actor_2d(state_img)
    x = x.view(x.size(0),-1)

    y =  self.actor_1d(state_pos)
    y = y.view(y.size(0),-1)

    merged = torch.cat((x,y),dim=1)
    merged = self.actor_merged(merged)
  
    merged = torch.Tensor(self.max_action) * torch.tanh(merged)
    return merged
```



#### **Critic Model**

```
  def __init__(self ):
        super(Critic, self).__init__()

**Defining the First Critic neural network**
    self.features1 = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 64 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(1,1))
    )

    self.linear_1 = nn.Linear(state_dim + action_dim,20)
    # self.bn_c1 = nn.BatchNorm1d(20)
    self.linear_11 = nn.Linear(20,50)
    # self.bn_c11 = nn.BatchNorm1d(50)
    self.linear_12 = nn.Linear(50,100)
    # self.bn_c12 = nn.BatchNorm1d(100)

    self.features2 = nn.Sequential(
        nn.Linear(64 + 100 ,200),
        # nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200,300 ),
        # nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300,75),
        nn.ReLU(),
        nn.Linear(75,1)
    )


```

**Defining the second Critic neural network**

         # Defining the second Critic neural network
        self.features3 = nn.Sequential(
            nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32 , 64 ,kernel_size = 3 ,stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
    
        self.linear_2 = nn.Linear(state_dim + action_dim,20)
        # self.bn_c2 = nn.BatchNorm1d(20)
        self.linear_21 = nn.Linear(20,50)
        # self.bn_c21 = nn.BatchNorm1d(50)
        self.linear_22 = nn.Linear(50,100)
        # self.bn_c22 = nn.BatchNorm1d(100)
    
        self.features4 = nn.Sequential(
            nn.Linear(64 + 100 ,200),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200,300 ),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300,75),
            nn.ReLU(),
            nn.Linear(75,1)
        )
    
      def forward(self, state_0,state_1, actor_action):
    
        # Forward-Propagation on the first Critic Neural Network
      
        y =  torch.cat((state_1,actor_action),dim=1)
    
        x1 = self.features1(state_0)
        x1 = x1.view(x1.size(0),-1)
    
        y1 = F.relu(self.linear_1(y))
        y1 = F.relu(self.linear_11(y1))
        y1 = F.relu(self.linear_12(y1))
        # y1 = F.relu(self.bn_c1(self.linear_1(y)))
        # y1 = F.relu(self.bn_c11(self.linear_11(y1)))
        # y1 = F.relu(self.bn_c12(self.linear_12(y1)))
        
        y1 = y1.view(y1.size(0),-1)
        merged_1 = torch.cat((x1,y1),dim=1)
        merged_1 = self.features2(merged_1)
    
        # Forward-Propagation on the second Critic Neural Network
        x2 = self.features3(state_0)
        x2 = x2.view(x2.size(0),-1)
    
        y2 = F.relu(self.linear_2(y))
        y2 = F.relu(self.linear_21(y2))
        y2 = F.relu(self.linear_22(y2))
    
        # y2 = F.relu(self.bn_c2(self.linear_2(y)))
        # y2 = F.relu(self.bn_c22(self.linear_22(y2)))
        y2 = y2.view(y2.size(0),-1)
    
        merged_2 = torch.cat((x2,y2),dim=1)
        merged_2 = self.features4(merged_2)
    
        return merged_1 , merged_2
    
      def Q1(self, state_0,state_1, actor_action):
       
        y =  torch.cat((state_1,actor_action),dim=1)
    
        x1 = self.features1(state_0)
        x1 = x1.view(x1.size(0),-1)
    
        y1 = F.relu(self.linear_1(y))
        y1 = F.relu(self.linear_11(y1))
        y1 = F.relu(self.linear_12(y1))
        # y1 = F.relu(self.bn_c1(self.linear_1(y)))
        # y1 = F.relu(self.bn_c11(self.linear_11(y1)))
        # y1 = F.relu(self.bn_c12(self.linear_12(y1)))
    
        y1 = y1.view(y1.size(0),-1)
    
        merged_1 = torch.cat((x1,y1),dim=1)
        merged_1 = self.features2(merged_1)
        return merged_1



**Car_moving_video** files shows the car moving in the city map.
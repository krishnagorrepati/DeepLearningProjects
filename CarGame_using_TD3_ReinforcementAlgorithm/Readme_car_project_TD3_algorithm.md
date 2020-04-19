## Reinforcement Learning (Car Game) Project using TD3 algorithm

Steps Followed:-

- Defined models for Actor and Critics and applied to TD3 Algorithm. Integrated the T3D training algorithm in TestGame.Update() function

**test.py file has the following details**

> - **TestGame class has the below functions**
>
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



**test.kv file has following objects to be implemented in test.py file**

> - **TestCar**
>
> - **TestGame**
>
>   

**TD3.py has the below obects:-**

> **ReplayBuffer(object): has below functions**
>
> - **add()  adds the transition [state, next_state, action, reward, done] to its internal storage**
> - **_init__()**
> - **sample() returns the batch of transitions**
>
> **TD3() class where TD3 algorithm is implemented.**



#### Actor and Critic Models are defined and T3D algorithm is implemented

**Actor Model**

  def __init__(self, action_dim, max_action):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32,32 ,kernel_size=3, stride=2)
    self.bn3 = nn.BatchNorm2d(32)
    self.conv4 = nn.AvgPool2d(4)
    self.layer_3 = nn.Linear(32, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = x.reshape(1,-1)
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x

#### **Critic Model**

  def __init__(self ):
        super(Critic, self).__init__()

**Defining the First Critic neural network**

​        self.features1 = nn.Sequential(
​        	nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
​        	nn.BatchNorm2d(16),
​        	nn.ReLU(),
​        	nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
​        	nn.BatchNorm2d(32),
​        	nn.ReLU(),
​        	nn.Conv2d(32 , 32 ,kernel_size = 3 ,stride = 2),
​        	nn.BatchNorm2d(32),
​        	nn.ReLU(),
​        	nn.AvgPool2d(4) 

​	self.features2 = nn.Sequential(
​       	 nn.Linear(32+1,300),
​        	nn.ReLU(),
​        	nn.Linear(300,400 ),
​        	nn.ReLU(),
​        	nn.Linear(400,1)
   	 )

**Defining the second Critic neural network**

      self.features3 = nn.Sequential(
        nn.Conv2d(1 , 16 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32 , 32 ,kernel_size = 3 ,stride = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(4) 
    )
    self.features4 = nn.Sequential(
            nn.Linear(32+1,300), # state dim + Action dim
            nn.ReLU(),
            nn.Linear(300,400 ),
            nn.ReLU(),
            nn.Linear(400,1)
        )
        
    def forward(self, x, y):
        # Forward-Propagation on the first Critic Neural Network
    
        x1 = self.features1(x)
        x1 = x1.view(x1.size(0),-1)
        y1 = y.view(y.size(0),-1)
        merged_1 = torch.cat((x1,y1),dim=1)
        merged_1 = self.features2(merged_1)
    
        # Forward-Propagation on the second Critic Neural Network
        x2 = self.features3(x)
        x2 = x2.view(x2.size(0),-1)
        y2 = y.view(y.size(0),-1)
        merged_2 = torch.cat((x2,y2),dim=1)
        merged_2 = self.features4(merged_2)
        return merged_1 , merged_2


**Current Problems being faced with my code:-**

The car screen is in freeze state. I am  trying to solve the issue.
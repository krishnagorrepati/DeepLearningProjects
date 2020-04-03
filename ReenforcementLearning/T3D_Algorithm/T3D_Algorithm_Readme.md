
1. **Environment —** Physical world in which the agent operates
2. **State —** Current situation of the agent
3. **Reward —** Feedback from the environment
4. **Policy —** Method to map agent’s state to actions
5. **Value —** Future reward that an agent would receive by taking an action in a particular state

[**Deep Q-Networks(DQNs)** ](https://deepmind.com/research/dqn/)which use Neural Networks to estimate Q-values. But DQNs can only handle discrete, low-dimensional action spaces. DQN cannot be applied to continuous action spaces. 

DPG uses a deterministic policy gradient. This induces overestimation error, which in turn leads to bias, as well as weak policy updates. TD3 addresses this issue by implementing a Clipped Double Q-Learning algorithm, which maintains and bounds two Q-functions.

[**Deep Deterministic Policy Gradient(DDPG)**](https://arxiv.org/abs/1509.02971) is a model-free, off-policy, actor-critic algorithm that tackles this problem by learning policies in high dimensional, continuous action spaces. The figure below is a representation of **actor-critic** architecture.

![img](https://miro.medium.com/max/1842/1*azzV78wFkRq9ePrzGnvf5Q.png)


### ASYNCHRONOUS ADVANTAGE ACTOR CRITIC ALGORITHM  

We build two separate networks for Actor and Critic. The actor is going to predict the action. But this time it's actions will have values and those values must be equal to the Critic's values! 
With one max Q value (Critic), the model may not get stabilized.

![A3C_Model](https://github.com/krishnagorrepati/DeepLearningProjects/blob/master/img10.png)

##  **T3D OR TWIN DELAYED DDPG** 

it uses 2 critics.

###  **STEP 1**  

Initialize the Experience Replay Memory, with a  max size of 1e6 (404). We will populate it with each new transition.

**ReplyBuffer<CurObservation,  CurAction,  Reward, nextObservation>**

 

### **STEP 2**



We build **TWO** kinds of actor models. One called the Actor Model and another called the Actor Target.


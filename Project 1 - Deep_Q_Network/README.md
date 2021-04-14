[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif 
"Trained Agent"


[image2]: layers_48x32_579ep.png  "im3_48x32_579ep"
[image3]: layers_80x88_572ep.png  "im4_80x88_572ep"
[image4]: layers_64x56_590ep.png  "im5_64x56_590ep"
[image5]: layers_80x88_633ep.png  "im6_80x88_633ep"

# Project 1: Navigation

### Introduction

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting 
a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while 
avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Getting Started
Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the environment.

Place the file in the DRLND GitHub repository, in the p1_navigation/ folder, and unzip (or decompress) the file.

Instructions
Follow the instructions in Navigation.ipynb to get started with training your own agent!

(Optional) Challenge: Learning from Pixels
After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place! In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction. A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment. This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view. (Note: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Windows (32-bit): click here
Windows (64-bit): click here
Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file. Next, open Navigation_Pixels.ipynb and follow the instructions to learn how to use the Python API to control the agent.

(For AWS) If you'd like to train the agent on AWS, you must follow the instructions to set up X Server, and then download the environment for the Linux operating system above.

### Project complietion criteria

To tackle the episodic task environment, the agent need to attain an average score of +13 
across 100 consecutive episodes.

### Environment

The simulated environment is assisted by Unity application _Banana_ lying in the subdirectory "_Banana_Windows_x86_64_".
The below is how we start the environment of this Deep_Q_network project.

_env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")_

### Sessions in the training phases.

We run several training sessions in according  to the variable _numb_of_trains_.
For each training session, the obtained weights are saved into the file such as _weights_0.trn_ and  _weights_1.trn_

For each training session, we run the *Deep-Q-Network* procedure **dqn** to construct the **agent** with different parametersas as follows:

  agent = **Agent**(state_size=37, action_size=4, seed=1, fc1_units=fc1_nodes, fc2_units=fc2_nodes)       
  scores, episodes = **dqn**(n_episodes = 2000, eps_start = epsilon_start, train_numb=i)  
  
### Employed parameters in the training phases.

We utilized three paremters to train and find the most effective model :  _fc1_units_, _fc2_units_,  _eps_start_.
At the end of each session these parameters together with the episode number (at which the training is finished) 
are saved into the corresponding lists. We use this parameterson the step of testing of weights.
For each training session, the random values of the tesing weight for each parameters are as follows:
 * _eps_start_ is set from 0.970 to 0.999 with step 0.001, 
 * _fc1_units_ is set from 48 to 128 with step 16,
 * _fc2_inits_ is set from fc1_units - 16 to fc1_units - 16 with step 8.

### Deep-Q-Network algorithm

the procedure f the _Deep-Q-Network_**dqn** manages the **double loop**. 
External loop (by _episodes_) is executed till the number of episodes reached the maximal number 
of episodes _n_episodes = 2000_ or the _completion criteria_ is executed.
The environment _env_  is reset with the parmeter _train_mode_=_True_.
For the completion criteria, we check  

  _np.mean(scores_window) >=13_,  

where _scores_window_ is the array of the type deque realizing  the shifting window of length <= 100.
The element _scores_window[i]_ contains the _score_ achieved by the algorithm on the episode _i_.


In the internal loop,  **dqn** gets the current _action_ from the **agent**.
By this _action_ **dqn** gets _state_ and _reward_ from Unity environment.
Then, the **agent** accept params _state,action,reward,next_state, done_
to the next training step. The variable _score_ accumulates obtained rewards.

### Agent

The class **Agent** is defined in _dqn_agent.py_. This is the well-known class implementing 
the following mechanisms:

* Two Q-Networks (local and target) using the simple neural network.
* Replay memory (using the class ReplayBuffer)
* Epsilon-greedy mechanism
* Q-learning, i.e., using the max value for all possible actions
* Computing the loss function by MSE loss
* Minimize the loss by gradient descend mechanism using the ADAM optimizer

### Model Q-Network

Both Q-Networks (local and target) are implemented by the class
**QNetwork** lying in the file _model.py_. This class implements the simple
neural network with 3 fully-connected layers and 2 
rectified nonlinear layers. This **QNetwork** is realized in the framework 
of package **PyTorch**. The number of neurons of the fully-connected layers are 
as follows:

 * Layer fc1,  number of neurons: _state_size_ x _fc1_units_, 
 * Layer fc2,  number of neurons: _fc1_units_ x _fc2_units_,
 * Layer fc3,  number of neurons: _fc2_units_ x _action_size_,
 
where _state_size_ = 37, _action_size_ = 8, _fc1_units_ and _fc2_units_
are the input params.
 
### Output of training

This is the typical output of training sessions:

fc1_units:  64 , fc2_units:  64
train_numb:  0 eps_start:  0.975
Episode: 538, elapsed: 0:10:51.311591, Avg.Score: 13.02,  score 19.0, How many scores >= 13: 59, eps.: 0.11
 terminating at episode : 538 ave reward reached +13 over 100 episodes    
 ![im2_96x88_585ep][image2]      
 
fc1_units:  64 , fc2_units:  72
train_numb:  1 eps_start:  0.974
Episode: 594, elapsed: 0:11:57.215037, Avg.Score: 13.01,  score 14.0, How many scores >= 13: 62, eps.: 0.09
 terminating at episode : 594 ave reward reached +13 over 100 episodes
 ![im3_48x32_579ep][image3]   
 
 fc1_units:  48 , fc2_units:  56
train_numb:  2 eps_start:  0.979
Episode: 578, elapsed: 0:11:36.222573, Avg.Score: 13.04,  score 13.0, How many scores >= 13: 56, eps.: 0.10
 terminating at episode : 578 ave reward reached +13 over 100 episodes
 ![im4_80x88_572ep][image4]   

 fc1_units:  64 , fc2_units:  56
train_numb:  3 eps_start:  0.995
Episode: 594, elapsed: 0:11:51.450521, Avg.Score: 13.06,  score 17.0, How many scores >= 13: 59, eps.: 0.09
 terminating at episode : 594 ave reward reached +13 over 100 episodes   
 ![im5_64x56_590ep][image5]   

 fc1_units:  112 , fc2_units:  96
train_numb:  4 eps_start:  0.975
Episode: 29, elapsed: 0:00:33.691869, Avg.Score: 0.24,  score 0.0, How many scores >= 13: 0, eps.: 0.878  


[Most of the code are accredited to Udacity's DQN cod]

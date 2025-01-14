
[//]: # (Image References)
[image1]:Tennis.gif  "Trained Agent"
[image2]:plot_3000episodes.png "Plot_3000"


# Project 3: Collaboration and Competition

### Introduction

In this project, we deal with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, where two agents control rackets to bounce ball over a net.     
If an agent hits a ball over net, the agent receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball   
out of bounds, the agent receives a reward of -0.01. Therefore, the goal of each agent is to maintain the ball in play.    
The observation space is 24-dimensional consisting of 8 variables corresponding to the position and velocity  
of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding 
to movement toward (or away from) the net, and jumping. The accompanying research paper can be found [here](https://arxiv.org/pdf/1706.02275.pdf).

![Trained Agent][image1]


### Maddpg Environment

The environment is simulated by Unity application _Tennis.app_ lying in the subdirectory _Tennis_Windows_x86_64_.
We start the environment as follows:

      env = UnityEnvironment(seed=seed, file_name="Tennis_Windows_x86_64/Tennis.app")
      
The task is episodic, and in order to solve the environment, the agents must get an **average score** of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents).       

Let us compare multi-agent environment to single agent environments. It requires the training of two separate agents, 
and the agents need to collaborate under certain situations (like don’t let the ball hit the ground) 
and compete under other situations (like gather as many points as possible). Just doing a simple extension 
of single agent RL by independently training the two agents does not work very well because the agents 
are independently updating their policies as learning progresses. And this causes the environment to appear 
non-stationary from the viewpoint of any one agent. 

### Prepare the environment on the local machine

You need at least the following three packages:

1. **deep-reinforcement-learning  (DRLND)**        
   The instructions to set up the DRLND repository can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). 
   This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

2. The project environment is similar to, but not identical to the _Tennis_ environment on the 
   [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).
   Instead this environment, the project works with the environment which is provided as a part of project
   (subdirectory 'python') 


3. **Unity environment _Tennis_**

    For this project, we not need to install Unity because the environment already built. The environment     
    can be downloaded as follows:

    - Windows (64-bit), [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)    
    - Windows (32-bit), [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)     
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

   Download this environment zip into  **p3_collab-compet/** folder, and unzip the file.

### Train the Agent

   Run the notebook _Tennis.ipynb_
   
   [1] import UnityEnvironment    
   [2] env = UnityEnvironment(seed=seed, file_name="Tennis_Windows_x86_64/Tennis.app")   # create environment        
   [3] Environments contain _brains_ which are responsible for deciding the actions of their associated agents.     
       We check for the first brain available.      
   [4] Examine the State and Action Spaces. We get the information frame as follows:   
       
     Number of agents: 2   
     Size of each action: 2   
     There are 2 agents. Each observes a state with length: 24    
     The state for the first agent looks like: 
     [ 0.          0.          0.          0.          0.          0.     
       0.          0.          0.          0.          0.          0.   
       0.          0.          0.          0.         -6.65278625 -1.5   
      -0.          0.          6.83172083  6.         -0.          0.        ]     
   
   [5]  Create _env_info_ and _maddpg agent_:

     env_info = env.reset(train_mode=True)[brain_name]      
     agent = maddpg_agent(num_agents=2, state_size=24, action_size=2)   

   [6]  Define and run the main function _train_ :
   
     scores_total, scores_global = train(maddpg, env, dir_chkpoints, n_episodes=3000)  
      
   [7]  Print graph of scores_total (blue bars) over all episodes, and  scores_global  
        (the line 'Avg on 100 episodes' - orange points)    
        The environment was solved in **1302 episodes**,  at this point the **Average Score** is achieved to **+0.5**,    
        see _Tennis.ipynb_ or _REPORT.ipynb_.   
        
        
### Train History

1. At **3000 episode** the **Average Score** is achived to **+1.14**.  
![Plot_3000][image2]


        
### Weights of the Trained Agent
  
  The **weights** of the trained agent are saved into files       
  
      checkpoint_actor_0.pth,  checkpoint_actor_1.pth,  checkpoint_critic_0.pth, checkpoint_critic_1.pth  
              
  stored in the directory 'dir_chk_3000d_episodes'

     
### Credit

Most of the code is based on Udacity's Mupti-agent DDPG code.

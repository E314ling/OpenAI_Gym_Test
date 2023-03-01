import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import gym

import tensorflow_probability as tfp


class experience_memory():

    def __init__(self, buffer_capacity, batch_size, state_dim, action_dim):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

class Pendulum():

    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        self.batch_size = 64
        self.max_memory_size = 1000000

        self.state_dim = 3
        self.action_dim = 1

        self.gamma = 1
        self.tau = 0.005

        # for the entropy regulization
        self.temp = 0.2

        self.lower_action_bound = -2
        self.upper_action_bound = 2

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        
    
        self.critic_1 = self.get_critic()
        self.target_critic_1 = self.get_critic()
        self.target_critic_1.set_weights(self.critic_1.get_weights())

        self.critic_2 = self.get_critic()
        self.target_critic_2 = self.get_critic()
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        self.actor = self.get_actor()
            
        self.critic_lr = 0.003
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)

        self.actor_lr = 0.003
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
   
        self.lr_decay = 0.9999

        self.update = 1
       
    def save_model(self):
        self.critic.save('./Models/critic.h5')
        
        self.actor.save('./Models/actor.h5')
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def get_critic(self):
      input_state = keras.Input(shape =(self.state_dim,))
      input_action = keras.Input(shape =(self.action_dim,))
      input = tf.concat([input_state, input_action], axis = 1 )
      d1 = layers.Dense(256, activation = 'relu')(input)
      d2 = layers.Dense(256, activation = 'relu')(d1)
      out = layers.Dense(1)(d2)
      model = keras.Model(inputs =  [input_state, input_action], outputs = out)
      return model

    
    def get_actor(self):
      input = keras.Input(shape = (self.state_dim,))

      d1 = layers.Dense(256, activation = 'relu')(input)
      d2 = layers.Dense(256, activation = 'relu')(d1)

      mu = layers.Dense(self.action_dim)(d2)
      log_std = layers.Dense(self.action_dim)(d2)

      model = keras.Model(inputs = input, outputs = [mu, log_std])
     
      return model
    
    def transform_actor(self, mu, log_std):
      clip_log_std = tf.clip_by_value(log_std, -20,2)

      std =  tf.exp(clip_log_std)
      
      dist = tfp.distributions.Normal(mu, std,allow_nan_stats=False)

      action_ = dist.sample()
      action = tf.tanh(action_)

      log_pi = dist.log_prob(action_)
      log_pi_a = log_pi - tf.reduce_sum(tf.math.log((1-action**2) + 1e-6), axis = 1, keepdims = True)
      action = self.upper_action_bound*action
      return action, log_pi_a

    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        mu, std = self.actor(next_state_batch)
        pi_a ,log_pi_a = self.transform_actor(mu, std)
                                              
        target_1 = self.target_critic_1([next_state_batch, pi_a])
        target_2 = self.target_critic_1([next_state_batch, pi_a])
        
        target_vals =  tf.minimum(target_1, target_2)

        # soft target
      
        y = tf.stop_gradient(reward_batch + (1-done_batch)* self.gamma*(target_vals - self.temp*log_pi_a))

        with tf.GradientTape() as tape1:
            
            critic_value_1 = self.critic_1([state_batch, action_batch])
            
            critic_1_loss = losses.MSE(y,critic_value_1 )

        with tf.GradientTape() as tape2:
            
            critic_value_2 = self.critic_2([state_batch, action_batch])
            
            critic_2_loss = losses.MSE(y,critic_value_2)

        
        critic_1_grad = tape1.gradient(critic_1_loss, self.critic_1.trainable_variables)   
        self.critic_optimizer_1.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))

        critic_2_grad = tape2.gradient(critic_2_loss, self.critic_2.trainable_variables)   
        self.critic_optimizer_2.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))


       

    @tf.function
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        
        with tf.GradientTape() as tape:
            mu,std = self.actor(state_batch)
            pi_a, log_pi_a = self.transform_actor(mu, std)
            critic_value_1 = self.critic_1([state_batch, pi_a])
            critic_value_2 = self.critic_2([state_batch, pi_a])
            min_critic = tf.minimum(critic_value_1, critic_value_2)

            soft_q = min_critic - self.temp * log_pi_a

            # for maximize add a minus '-tf.math.reduce_mean(soft_q)'
            actor_loss = -tf.math.reduce_mean(soft_q)
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
     
    def learn(self,episode):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if (episode % self.update == 0 and episode != 0):
            self.update_actor(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

MC = Pendulum()

ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
num_episode = 500
decay = 0.9999
for ep in range(num_episode):
    
    done = False
    state,_ = MC.env.reset()
    state = tf.expand_dims(tf.convert_to_tensor(state),0)
    
    #state = np.reshape(state, [1,MC.state_dim])
    episodic_reward = 0
    t_counter = 0

    #if (ep % 100 == 0 and ep != 0):
        #MC.run_MC()
    while(True):
        mu, log_std = MC.actor(state)
        action_,_ = MC.transform_actor(mu, log_std)
        
        action = action_[0].numpy()
       
        new_state, reward, done,_, info = MC.env.step(action)
        
        new_state = tf.expand_dims(tf.convert_to_tensor(new_state.reshape(MC.state_dim)),0)
       
        #new_state = np.reshape(new_state, [1,MC.state_dim])
        episodic_reward += reward
        
        MC.buffer.record((state,action,reward, new_state, done))
       
        MC.learn(ep)
    
        MC.update_target(MC.target_critic_1.variables, MC.critic_1.variables)
                    
        MC.update_target(MC.target_critic_2.variables, MC.critic_2.variables)
       

        state = new_state
        t_counter +=1
        if (done):
            break
    ep_reward_list.append(episodic_reward)
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-20:])
    print("Episode * {} * AVG Reward is ==> {}, actor_lr ==> {}".format(ep, avg_reward,MC.actor_lr))
    avg_reward_list.append(avg_reward)
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
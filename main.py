import os
import tensorflow as tf
import numpy as np
import gym
import time

LOG_DIR = os.path.join(os.getcwd(), "log3")
slim = tf.contrib.slim

class Agent:
    def __init__(self, session, num_states, num_actions, epsilon_initial, epsilon_end):
        self.sess = session
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon_initial = epsilon_initial
        self.epsilon_end = epsilon_end

        self.build_prediction_net()
        self.build_train_op()
        self.summary = tf.summary.merge_all()
    
    def build_prediction_net(self):
        self.state_placeholder = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)

        with tf.variable_scope("prediction_net"):
            hidden0 = slim.fully_connected(self.state_placeholder, 16)
            hidden1 = slim.fully_connected(hidden0, 16)
            self.output = slim.fully_connected(hidden1, self.num_actions, activation_fn=tf.nn.softmax)

    def predict_action_logit(self, state):
        action = self.sess.run(self.output, feed_dict={self.state_placeholder: state})
        return action
    
    def epsilon_greedy_action(self, action_logits, epsilon_progress):
        # epsilon_progress should start from 1.0 to 0.0
        epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * epsilon_progress
        if np.random.random() < epsilon:
            action_index = np.argmax(np.random.random(action_logits.shape))
        else:
            action_index = np.argmax(action_logits)
        
        return action_index

    def predict_action_with_epsilon(self, state, epsilon):
        # input one state
        action_logit = self.predict_action_logit([state])
        return self.epsilon_greedy_action(action_logit[0], epsilon)

    def build_train_op(self):
        self.action_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.reward_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)

        logit_flat = tf.reshape(self.output, [-1])

        # Index to pickup largest logit from flattened array
        index = tf.range(tf.shape(self.output)[0]) * tf.shape(self.output)[1]
        index = index + self.action_placeholder

        selected_logits = tf.gather(logit_flat, index)

        self.loss = -tf.reduce_mean(tf.log(selected_logits) * self.reward_placeholder)

        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)

    def train(self, states, actions, discounted_rewards):
        feed_dict = {
            self.state_placeholder: states,
            self.action_placeholder: actions,
            self.reward_placeholder: discounted_rewards
        }

        _, loss, sumury = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
        
        return loss, sumury

class EpisodeMemory:
    def __init__(self, not_allow_discount_return_calculation=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.discounted_returns = []

        self.discounted_returns_calculated = not_allow_discount_return_calculation
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.discounted_returns = []
    
    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.states_next += episode.states_next
        self.discounted_returns += episode.discounted_returns

    def add_one_step(self, state, action, reward, state_next):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_next.append(state_next)

    def calculate_discounted_returns(self, discount_rate=0.98):
        if self.discounted_returns_calculated:
            assert "Wrong operation"

        self.discounted_returns = [0 for _ in range(len(self.rewards))]
        self.discounted_returns[-1] = self.rewards[-1]

        for i in range(len(self.rewards) - 2, -1, -1):
            self.discounted_returns[i] = self.rewards[i] + self.discounted_returns[i + 1] * discount_rate
        
        self.discounted_returns_calculated = True

def main():
    env = gym.make("CartPole-v0")
    num_states = env.observation_space.shape[0]  # 4 for CartPole-v0
    num_actions = env.action_space.n  # 2 for CartPole-v0

    epsilon_initial = 0.7
    epsilon_end = 0.004
    epsilon_decay = 0.999

    max_episode = 100000
    max_itr_one_episode = 700
    train_frequency = 6

    with tf.Session() as sess:
        agent = Agent(sess, num_states, num_actions, epsilon_initial, epsilon_end)

        # Variables to save/restore
        global_step_variable = tf.Variable(0, trainable=False, name="global_step")
        episode_proceeded_count_variable = tf.Variable(0, trainable=False, name="episode_proceeded_count")

        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load checkpoint " + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initialize variables")
            sess.run(tf.global_variables_initializer())
        
        global_step = sess.run(global_step_variable)
        episode_proceeded_count = sess.run(episode_proceeded_count_variable)

        global_memory = EpisodeMemory(True)

        # Initial epsilon calculation
        epsilon = 1.0
        for i in range(episode_proceeded_count):
            epsilon *= epsilon_decay

        max_local_episode = 0
        start_time = time.time()
        
        for episode_counter in range(episode_proceeded_count, max_episode):
            
            local_memory = EpisodeMemory()

            state = env.reset()
            episode_reward = 0.0
            
            for local_episode_counter in range(max_itr_one_episode):
                # Predict action index from state
                action = agent.predict_action_with_epsilon(state, epsilon)
                state_next, reward, terminal, _ = env.step(action)
                
                if local_episode_counter > 50 and local_episode_counter <= 100:
                    reward += 0.2
                elif local_episode_counter > 100 and local_episode_counter <= 150:
                    reward += 0.5
                elif local_episode_counter > 150 and local_episode_counter <= 200:
                    reward += 1.2
                elif local_episode_counter > 200 and local_episode_counter <= 400:
                    reward += 2.0
                elif local_episode_counter > 400:
                    reward += 3.5

                reward = reward + reward * (local_episode_counter / 100)
                
                if (terminal):
                    reward = -10.0

                local_memory.add_one_step(state, action, reward, state_next)

                state = state_next
                episode_reward += reward
                global_step += 1

                if terminal:
                    local_memory.calculate_discounted_returns()
                    global_memory.add_episode(local_memory)

                    if local_episode_counter > max_local_episode:
                        max_local_episode = local_episode_counter

                    if (episode_counter % train_frequency) == 0:
                        loss, sumury = agent.train(np.array(global_memory.states), np.array(global_memory.actions), np.array(global_memory.discounted_returns))

                        current_time = time.time()
                        print("Ep " + format(episode_counter, "4d") + ", loss " + format(loss, ".4f") + ", epsilon " + format(epsilon, ".3f") + ", maxLocalEp " + str(max_local_episode) + ", epReward " + format(episode_reward, ".1f") + ", elapse " + format((current_time - start_time), ".1f") + "[s]")
                        start_time = current_time

                        max_local_episode = 0

                        # Variables to save
                        sess.run(global_step_variable.assign(global_step))
                        sess.run(episode_proceeded_count_variable.assign(episode_counter))

                        summary_writer.add_summary(sumury, episode_counter)
                        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step = episode_counter)

                        global_memory.reset()
                    
                    break
            
            # One episode loop end
            epsilon *= epsilon_decay


def predict():
    env = gym.make("CartPole-v0")
    num_states = env.observation_space.shape[0]  # 4 for CartPole-v0
    num_actions = env.action_space.n  # 2 for CartPole-v0

    with tf.Session() as sess:
        agent = Agent(sess, num_states, num_actions, epsilon_initial=5.0, epsilon_end=0.001)

        # Variables to save/restore
        global_step_variable = tf.Variable(0, trainable=False, name="global_step")
        episode_proceeded_count_variable = tf.Variable(0, trainable=False, name="episode_proceeded_count")

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load checkpoint " + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            #print("Initialize variables")
            #sess.run(tf.global_variables_initializer())
            print("No saved variables")
            return

        global_step = sess.run(global_step_variable)
        episode_proceeded_count = sess.run(episode_proceeded_count_variable)
        print("Episode " + str(episode_proceeded_count))

        while True:
            state = env.reset()

            for i in range(1000):
                env.render()
                action = agent.predict_action_with_epsilon(state, epsilon=0.0)
                state_next, reward, terminal, _ = env.step(action)

                state = state_next

                if terminal:
                    print("Reward " + str(reward) + ", itr " + str(i))
                    break

if __name__ == "__main__":
    #main()
    predict()



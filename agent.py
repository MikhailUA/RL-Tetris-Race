import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import warnings
import random as rnd
import math
warnings.filterwarnings("ignore")

global team_name, folder, env_name
team_name = 'ml_team # N4' # TODO: change your team name
folder = 'tetris_race_qlearning'
env_name = 'TetrisRace-v0' # do not change this


class TetrisRaceQLearningAgent:
    def __init__(self,env,learning_rate = 0.5, discount_factor =0.5,
                 exploration_rate =0.5, exploration_decay_rate =0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.actions = env.unwrapped.actions
        self._num_actions = len(self.actions)
        self.state = None
        self.action = None

        # =============== TODO: Your code here ===============
        #  We'll use tabular Q-Learning in our agent, which means
        #  we need to allocate a Q - Table.Think about what exactly
        #  can be represented as states and how many states should
        #  be at all. Besides this remember about actions, which our
        #  agent will do. Q - table must contain notion about the state,
        #  represented as one single integer value (simplest option) and weights
        #  of each action, which agent can do in current env.

        self.wall_iterator = env.unwrapped.wall_iterator # passed walls counter
        
        self.screen_width = env.unwrapped.screen_width
        self.n_bins = 100
        self.x_space = np.linspace(0, 400, self.n_bins)
        self.q_table = self.init_q_table()
        self.maxT = 1

    def choose_action(self, observation):
        # =============== TODO: Your code here ===============
        #  Here agent must choose action on each step, 
        #  solving exploration-exploitation trade-off. 

        #  Remember that in general exploration rate is responsible for
        #  agent behavior in unknown world conditions and main motivation is explore world.
        #  Exploitation rate - choose already known actions and moving through known states.
        #  Think about right proportion that parameters for better solution
        


        state_id = int(observation[1]) *-1
        x = int(observation[0])
        x = np.digitize(x, self.x_space)
        #print(state_id)
        if (self.is_state_exists(state_id) == False):
            self.add_new_state(state_id)

        state = self.get_state(state_id)

        action = self.ACTION_LEFT()
        # idx 1 left, idx 2 right
        
        value_right = self.get_state_action_value(state, x, self.ACTION_RIGHT())
        value_left  = self.get_state_action_value(state, x, self.ACTION_LEFT())

        if (self.maxT < state_id):
            self.maxT = state_id

        e = 1 - 1 / (1 + math.exp(-state_id / 80) + 3)
        e = state_id / (self.maxT) + 0.3
        #print(e)

        e_rand = rnd.randint(0, 100) / 100

 
        greedy = True
        if (value_right == value_left):
            action = rnd.randint(self.ACTION_LEFT(), self.ACTION_RIGHT())
            #action = self.ACTION_LEFT()
        elif (value_right > value_left): # select action with max value
            action = self.ACTION_RIGHT()
         
        if (e < e_rand):
            greedy = False

        #print(greedy)
        if (greedy == False and action == self.ACTION_RIGHT() and value_right > value_left):
            #print(action)
            action == self.ACTION_LEFT()
            #print(action)
        elif(greedy == False and action == self.ACTION_LEFT() and value_right < value_left):
            action == self.ACTION_RIGHT()

        return action 

    def update_q_table(self, observation, action, reward, observation_):
        # =============== TODO: Your code here ===============
        #  Here agent takes action('moves' somewhere), knowing
        #  the value of Q - table, corresponds current state.
        #  Also in each step agent should note that current
        #  'Q-value' can become 'better' or 'worsen'. So,
        #   an agent can update knowledge about env, updating Q-table.
        #   Remember that agent should choose max of Q-value in  each step
        
        new_state_id = int(observation_[1]) *-1
        state_id = int(observation[1]) *-1

        new_state_id_x = int(observation_[0])
        state_id_x = int(observation[0])

        new_state_id_x = np.digitize(new_state_id_x, self.x_space)
        state_id_x = np.digitize(state_id_x, self.x_space)

        if (self.is_state_exists(new_state_id) == False):
            self.add_new_state(new_state_id)

        qSA  = self.get_state(state_id) # get state values for current state        
        qSA_ = self.get_state(new_state_id) # get state values for next state

        action_value = self.get_state_action_value(qSA, state_id_x, action)

        max_action_value_of_new_state = max(
            self.get_state_action_value(qSA_, new_state_id_x, self.ACTION_LEFT()), 
            self.get_state_action_value(qSA_, new_state_id_x, self.ACTION_RIGHT())
            )

        lr = (1 - state_id / (self.maxT + 0.0001) ) * self.learning_rate / 100
        #print(state_id)
        #print(lr)

        action_value_new = action_value + lr * (reward + self.discount_factor / 100 * max_action_value_of_new_state - action_value)

        self.update_state_value(state_id, state_id_x, action, action_value_new)  

        return self.q_table

#==========================================================================================================
#==========================================================================================================
#==========================================================================================================
        


    def init_q_table(self):  # updated
        return np.empty((0, self.n_bins, 2))

    def ACTION_LEFT(self):
        return  0

    def ACTION_RIGHT(self):
        return  1

    def is_state_exists(self, state_id): # updated
        q_tbl = self.q_table
        if (len(q_tbl) - 1 >= state_id):
            return True
        return False

    def add_new_state(self, state_id): # updated
        if (self.is_state_exists(state_id)):
            raise Exception("State already exists")
        self.q_table = np.vstack((self.q_table, [np.zeros((self.n_bins,2))] ))

    def get_state(self, state_id): # updated
        if (self.is_state_exists(state_id) == False):
            raise Exception("State does not exists")
        return self.q_table[state_id]

    def update_state_value(self, state_id, x, action, value):   # updated      
        self.q_table[state_id, x, action] = value

    def get_state_action_value(self, state, x, action): # updated
        return state[x, action]


class EpisodeHistory:
    def __init__(self,env,
                 learn_rate,
                 discount,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100
                 ):
        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.lr = learn_rate
        self.df = discount

        self.lvl_step = env.unwrapped.walls_per_level
        self.lvl_num = env.unwrapped.levels
        self.difficulty = env.unwrapped.level_difficulty
        self.point_plot = None
        self.mean_plot = None
        self.level_plots = []
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(3, 3), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History. Team {}".format(team_name))

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)
        self.ax.set_title("Episode Length History (lr {}, df {})".format(self.lr, self.df))
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")
        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")
        for i in range(0, self.lvl_num):
            self.level_plots.append(plt.plot([],[], linewidth =1.0, c="#207232",ls ='--'))

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update levels plots
        for i in range(1, self.lvl_num+1):
            xl = range(plot_left_edge, plot_right_edge)
            yl = np.zeros(len(xl))
            yl[:] = i * self.lvl_step
            cur_lvl_curve = self.level_plots[i - 1][0]
            cur_lvl_curve.set_xdata(xl)
            cur_lvl_curve.set_ydata(yl)
            self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        mean_kernel_size = 101
        rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        rolling_means = pd.Series(
            rolling_mean_data).rolling(
            window=mean_kernel_size,
            min_periods=0).mean()[mean_kernel_size:]
        self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.001)

    def is_goal_reached(self, episode_index):
        ''' DO NOT CHANGE THIS FUNCTION CODE.'''
        # From here agent will receive sygnal about end of learning
        arr = self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1]
        avg = np.average(arr)
        if self.difficulty == 'Easy':
            answer = avg >= self.goal_avg_episode_length + 0.5
        elif len(arr)>0:
            density = 2 * np.max(arr) * np.min(arr) / (np.max(arr) + np.min(arr))
            answer = avg >= self.goal_avg_episode_length + 0.5 and density >= avg

        return answer


class Controler:
    def __init__(self, parent_mode = True , episodes_num = 10000, global_env = []):
        self.team_name =  team_name
        self.exp_dir = folder + '/' + self.team_name
        random_state = 0
        self.agent_history = []
        self.history_f = True
        self.learning_rate = 100
        self.discount_factor = 100

        self.window = 50

        if parent_mode == False:
            # ====== TODO: your code here======
            # To run env with different parameters you can use another values of named variables, such as:
            #
            # walls_num = x --> number of obstacles (walls). The number must be  x > 6 and x % 3 == 0
            # walls_spread = x --> distance between walls. Too small value leads to no solution situation
            # episodes_to_run = x --> number of agent's tries to learn
            # world_type = 'Fat' or 'Thin' --> make objects more thicker or thinner
            # smooth_car_step = 5 -->  smoothness of car moves (value of step by x)
            # level_difficulty ='Easy' or 'Medium' --> change number of bricks in walls
            # car_spawn = 'Random' or 'Center' --> place, where car starts go
            #
            # EXAMPLE:
            # env.__init__(walls_num = 6, walls_spread = 3, episodes_to_run = episodes_num)
            #
            # Best choice will try any of this different options for better understanding and
            # optimizing the solution.

            env = gym.make(env_name)
            env.__init__(episodes_to_run = episodes_num)
            env.seed(random_state)
            np.random.seed(random_state)
            lr = self.learning_rate
            df = self.discount_factor
            exr = 10
            exrd = 10
            print(env.screen_width)
            self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                            video_callable=self.video_callable)
            episode_history, end_index = self.run_agent(self, lr, df, exr, exrd, self.env,
                                                        verbose=False)
        else:
            # Here all data about env will received from main script, so
            # each team will work with equal initial conditions
            # ====== TODO: your code here======
            env = global_env
            env.seed(random_state)
            np.random.seed(random_state)

            self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                            video_callable=self.video_callable)
            episode_history, end_index = self.run_agent(self, self.learning_rate, self.discount_factor,
                                                        self.exploration_rate, self.exploration_decay_rate,
                                                        self.env, verbose=False)

    def run_agent(self, rate, factor, exploration, exp_decay, env, verbose=False):
        max_episodes_to_run = env.unwrapped.total_episodes
        #max_episodes_to_run = 10000
        max_timesteps_per_episode = env.unwrapped.walls_num

        goal_avg_episode_length = env.unwrapped.walls_num
        wall_coef = 6 / env.unwrapped.walls_num
        goal_consecutive_episodes = int(wall_coef * self.window)  # how many times agent can consecutive run succesful

        plot_episode_count = 200
        plot_redraw_frequency = 10

        # =============== TODO: Your code here ===============
        #   Create a Q-Learning agent with proper parameters.
        #   Think about what learning rate and discount factor
        #   would be reasonable in this environment.

        agent = TetrisRaceQLearningAgent(env,
                                         learning_rate=rate,
                                         discount_factor=factor,
                                         exploration_rate=exploration,
                                         exploration_decay_rate=exp_decay
                                         )
        # ====================================================
        episode_history = EpisodeHistory(env,
                                         learn_rate=rate,
                                         discount=factor,
                                         capacity=max_episodes_to_run,
                                         plot_episode_count=plot_episode_count,
                                         max_timesteps_per_episode=max_timesteps_per_episode,
                                         goal_avg_episode_length=goal_avg_episode_length,
                                         goal_consecutive_episodes=goal_consecutive_episodes)
        episode_history.create_plot()

        finish_freq = [0.5, True]  # desired percent finishes in window, flag to run subtask once

        startX = np.zeros((500))
        tmax = 0
        for episode_index in range(0, max_episodes_to_run):
            timestep_index = 0
            observation = env.reset()
           

            startX[int(observation[0])] += 1

            while True:
                t = observation[1]
                if (t < tmax):
                    tmax = t
                    print(tmax)

                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)  # Perform the action and observe the new state.

                if verbose == True:
                    env.render()
                    # self.log_timestep(timestep_index, action, reward, observation)

                if done and timestep_index < max_timesteps_per_episode - 1:
                    reward = -max_episodes_to_run

                QDF = agent.update_q_table(observation, action, reward, observation_)
                observation = observation_

                if done:
                    self.done_manager(self, episode_index, [], [], 'D')
                    if self.done_manager(self, episode_index, [], finish_freq, 'S') and finish_freq[1]:
                        foo = Classification()
                        finish_freq[1] = False

                    episode_history[episode_index] = timestep_index + 1
                    if verbose or episode_index % plot_redraw_frequency == 0:
                        episode_history.update_plot(episode_index)

                    if episode_history.is_goal_reached(episode_index):
                        print("Goal reached after {} episodes!".format(episode_index + 1))
                        end_index = episode_index + 1
                        foo = Regression(QDF)
                        self.done_manager(self, [], plt, [], 'P')

                        return episode_history, end_index
                    break
                elif env.unwrapped.wall_iterator - timestep_index > 1:
                    timestep_index += 1
        print("Goal not reached after {} episodes.".format(max_episodes_to_run))
        end_index = max_episodes_to_run
        print(np.where(startX > 0))

        x = agent.q_table.shape[0]
        y = agent.q_table.shape[1]
        z = agent.q_table.shape[2]


        np.savetxt("q_table.csv", np.reshape(agent.q_table, (x,y*z)) , delimiter=",")
        return episode_history, end_index

    def done_manager(self, episode_ind, plt, top, mode):
        # Call this function to handle episode end event and for storing some
        # result files, pictures etc

        if mode == 'D':  # work with history data
            refresh_each = 1000
            self.agent_history.append(self.env.unwrapped.wall_iterator)
            if episode_ind % refresh_each == 0 and self.history_f:
                root = self.exp_dir.split('/')[0]
                base = '/_data'
                path = root + base
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(path + '/' + self.team_name + '.pickle', 'wb') as f:
                    pickle.dump(self.agent_history, f)
        if mode == 'P':  # work woth progress plot
            path = self.exp_dir + '/learn_curve'
            name = '/W ' + str(self.env.unwrapped.walls_num) + \
                   '_LR ' + str(self.learning_rate) + '_DF ' + str(self.discount_factor)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name + '.png')
        if mode == 'S':  # call subtasks when condition
            if episode_ind > self.window:
                arr = self.agent_history[episode_ind - self.window: episode_ind]
                mx = np.max(arr)
                ind = np.where(arr == mx)[0]
                count = ind.shape[0]
                prc = count / self.window if mx > self.env.unwrapped.walls_per_level * 2  else 0
                x = self.agent_history
                total_finishes = sum(map(lambda x: x > self.env.unwrapped.walls_per_level * 2, x))

                return prc >= top[0] and total_finishes > 100

    def video_callable(episode_id):
        # call agent draw each N episodes
        return episode_id % 1000 == 0 or episode_id == 8999

    def log_timestep(self, index, action, reward, observation):
        # print parameters in console
        format_string = "   ".join(['Timestep:{}',
                                    'Action:{}',
                                    'Reward:{}',
                                    'Car pos:{}',
                                    'WallY pos:{}'])
        print('Timestep: format string ', format_string.format(index, action, reward,
                                                               observation[0], observation[1]))

    def save_history(self, history, experiment_dir):
        # Save the episode lengths to CSV.
        filename = os.path.join(experiment_dir, "episode_history.csv")
        dataframe = pd.DataFrame(history.lengths, columns=["length"])
        dataframe.to_csv(filename, header=True, index_label="episode")


class Regression:
    def __init__(self, dataset, dependent_feature='DepDelay'):
        # =============== TODO: Your code here ===============
        # One of subtask. Receives dataset, must return prediction vector
        # DepDelay - flight departure delay. You should predict departure delay depending on other features.
        # You should add the code that will:
        #   - read dataset from file
        pass

    def coolMethodThatWillDoAllTheNeededStuffWithDataset(self):
        # =============== TODO: Your code here ===============
        # This is the method that will prepare dataset
        # You should add the code that will:
        #   - clean dataset from useless features
        #   - split data set to train and test parts
        #   - implement lable encoding and one hot encoding if needed
        #   - create regression model and fit it
        #   - make prediction values of dependant feature
        #   - calculate r2_score to check prediction accuracy
        #   - save the model
        #   - return r2_score, modified dataset, predicted values vector, saved regression model
        print('Hey, sexy mama, wanna kill all humans?')



class Classification:
    def __init__(self):
        # =============== TODO: Your code here ===============
        # One of subtask. Receives dataset, must return prediction vector
        # You should add the code that will:
        #   - read dataset from file
        pass

    def coolMethodThatWillDoAllTheNeededStuffWithDataset(self):
        #   - separate SalaryNormalized into two (0 and 1) classes by median
        #   - join all text features into one
        #   - make vectorization using sklearn.countVectoriser or any other vectorizer
        #   - fit classification model
        #   - calculate precision, recall and f1-score
        #   - save the model
        #   - return f1-score, modified dataset, saved classification model
        # #

        print('Kill all humans, kill all humans, must kill all humans...')
        pass


def main(env, parent_mode = True):
    obj = Controler
    obj.__init__(obj, parent_mode= parent_mode, global_env= env,episodes_num = 9000)

if __name__ == "__main__":
    if 'master.py' not in os.listdir('.'):
        main([],parent_mode=False)
    else:
        main(env,parent_mode)

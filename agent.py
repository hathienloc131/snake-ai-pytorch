import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import wandb
wandb.init(project="snake-ai-pytorch")

MAX_MEMORY = 1_000_000
BATCH_SIZE = 1024
LR = 0.0001
MAX_VIEW = 8
class Agent:

    def __init__(self, episodes):
        self.n_games = 0
        self.n_steps = 0
        self.max_step = 500000
        self.epsilon = 1.0 # randomness
        self.min_epsilon = 0.05 # no randomness
        self.epsilon_decay = 0.9995 # epsilon decay rate
        self.gamma = 0.95 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        self.trainer = QTrainer(lr=LR, gamma=self.gamma, episodes=episodes)


    def get_state(self, game):
        head = game.snake[0]
        point_l = [Point(head.x - 20 * i, head.y) for i in range(MAX_VIEW)]
        point_r = [Point(head.x + 20 * i, head.y) for i in range(MAX_VIEW)]
        point_u = [Point(head.x, head.y - 20 * i) for i in range(MAX_VIEW)]
        point_d = [Point(head.x, head.y + 20 * i) for i in range(MAX_VIEW)]
        
        point_ul = [Point(head.x - 20 * i, head.y - 20 * i) for i in range(MAX_VIEW)]
        point_ur = [Point(head.x + 20 * i, head.y - 20 * i) for i in range(MAX_VIEW)]
        point_dl = [Point(head.x - 20 * i, head.y + 20 * i) for i in range(MAX_VIEW)]
        point_dr = [Point(head.x + 20 * i, head.y + 20 * i) for i in range(MAX_VIEW)]
        total_point_view = [point_l, point_r, point_u, point_d, point_ul, point_ur, point_dl, point_dr]
        point_view = []
        for point_dir in total_point_view:
            current_point_view = []
            # current_idx = 0
            for idx, point in enumerate(point_dir):
                collision = game.is_collision(point)
                current_point_view.append(collision)
                if collision:
                    current_point_view.extend([1] * (MAX_VIEW - len(current_point_view)))
                    break
            point_view.extend(current_point_view)
            # point_view.append(current_idx * 1.0 / MAX_VIEW)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # 8*8 + 4 + 4 = 16
        state = [

            *point_view,

            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # head location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if self.trainer.learn_step_counter % self.trainer.TARGET_REPLACE_FREQ == 0:
            self.trainer.target_net.load_state_dict(self.trainer.evaluate_net.state_dict())
        self.trainer.learn_step_counter += 1
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss, lr_now = self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
        return loss, lr_now

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0,0,0]
        if np.random.rand() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.trainer.evaluate_net(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    episodes = 10000000
    max_step = 1500
    total_score = 0
    record = 0
    agent = Agent(episodes)
    game = SnakeGameAI()
    total_reward = 0
    log_step = 25
    total_loss = 0
    for i in range(episodes):
        for t in range(max_step):
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)
            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            total_reward += reward
            state_new = agent.get_state(game)
            # train short memory
            # agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
            

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                total_score += score
                if score > record:
                    record = score
                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon *= agent.epsilon_decay

                loss, lr_now = agent.train_long_memory()
                total_loss += loss
                agent.trainer.evaluate_net.save("model_latest.pth")
                if agent.n_games % log_step == 0:
                    wandb.log({"Mean Score": total_score/log_step, 'Record': record, "Mean Reward": total_reward/log_step, "Epsilon": agent.epsilon, "Episode": agent.n_games, "Loss": total_loss/log_step, "lr": lr_now})
                    total_reward = 0
                    total_score = 0
                    total_loss = 0
                break


if __name__ == '__main__':
    train()
import gym
import numpy as np
import pickle


# HYPER PARAMS
save_path = './skiing_records'
env_name = "Skiing-v0"
interval = 1


if __name__ == '__main__':
  import curses
  screen = curses.initscr()
  curses.noecho()
  curses.cbreak()
  screen.keypad(True)

  env = gym.make(env_name)
  obs = env.reset()

  score = 0.
  step = 0
  action = 0
  done = False

  saver = dict()
  saver['snapshots'] = []
  saver['scores'] = []
  saver['observations'] = []
  saver['actions'] = []

  while not done:
    env.render()
    if step % interval == 0:
      saver['snapshots'].append(env.env.clone_full_state())
      saver['observations'].append(obs)
      saver['scores'].append(score)
      saver['actions'].append(action)

    # key pairing might be changed
    k = screen.getch()
    if k==curses.KEY_RIGHT:
      action = 1
    elif k==curses.KEY_LEFT:
      action = 2
    else:
      action = 0
    obs, reward, done, info = env.step(action)

    score += reward
    step += 1
  saver['snapshots'].append(env.env.clone_full_state())
  saver['observations'].append(obs)
  saver['scores'].append(score)
  saver['actions'].append(action)

  with open(save_path, 'wb') as f:
    pickle.dump(saver, f)

  env.close()

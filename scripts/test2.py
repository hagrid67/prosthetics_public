import multiprocessing
import uuid
from osim.env import ProstheticsEnv

def worker(worker_id):
  env = ProstheticsEnv(visualize=True)
  observation = env.reset()
  for i in range(200):
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      print("Step : {} + Reward : {} + worker_id : {}".format(i,reward, worker_id))
      print("Action : ", action)
      if done:
          print("RESET")
  env.reset()

jobs = []
for i in range(4):
  p = multiprocessing.Process(target=worker, args=(i,))
  jobs.append(p)
  p.start()

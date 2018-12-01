
import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "50fb66bb6e5f4fdac76d1cb68a5b9038"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token)

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)



while True:
    #observation, reward, done, info = env.step(env.action_space.sample())
    #[observation, reward, done, info] = client.env_step(my_controller(observation), True)
    [observation, reward, done, info] = client.env_step(env.action_space.sample(), True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()



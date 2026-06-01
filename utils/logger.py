import matplotlib.pyplot as plt

class TrainingLogger:
    def __init__(self):
        self.rewards = []
        self.steps = []

    def log(self, reward, steps):
        self.rewards.append(reward)
        self.steps.append(steps)

    def plot(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title("Reward per Episode")
        plt.subplot(1, 2, 2)
        plt.plot(self.steps)
        plt.title("Steps per Episode")
        plt.show()
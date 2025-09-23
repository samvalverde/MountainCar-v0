import matplotlib.pyplot as plt
import os

def plot_fitness(best, avg, out_dir="src/experiments/outputs", filename="training_curve.png"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(best, label="Best fitness")
    plt.plot(avg, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (total reward)")
    plt.legend()
    plt.title("Genetic Algorithm on MountainCar-v0")
    path = os.path.join(out_dir, filename)
    plt.savefig(path)
    plt.close()
    
    print(f"\nCurva de fitness guardada en {path}")

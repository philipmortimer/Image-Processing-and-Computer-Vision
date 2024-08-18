import matplotlib.pyplot as plt

# Code used to plot results for part 1a

#  FPR and TPR w.r.t classifier stage
fpr = [1, 0.0603282, 0.00877532]
tpr = [1, 1, 1]
stages = [0, 1, 2]

# Plots graph
plt.title("Classifier Performance across Training Stages")
plt.xlabel("Training Stage")
plt.ylabel("Score")
plt.xticks(range(len(stages)), stages)
plt.plot(tpr, label="TPR", marker='o')
plt.plot(fpr, label="FPR", marker='o')
plt.legend()
plt.savefig("Classifier Performance.png")
plt.show()
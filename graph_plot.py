import numpy as np
import matplotlib.pyplot as plt

mse_loss = np.asarray([0.0, 0.0001478, 0.0007683, 0.0009607,
                       0.0010790, 0.0013952, 0.0018208])

laplace_gender = np.asarray([0.972, 0.972, 0.970, 0.969, 0.969, 0.969, 0.968])

laplace_smile = np.asarray([0.977, 0.977, 0.976, 0.976, 0.976, 0.976, 0.975])

uniform_gender = np.asarray([0.972, 0.972, 0.970, 0.970, 0.970, 0.970, 0.970])

uniform_smile = np.asarray([0.977, 0.976, 0.976, 0.976, 0.976, 0.976, 0.976])

GAP_gender = np.asarray([0.972, 0.628, 0.547, 0.521, 0.518, 0.519, 0.503])

GAP_smile = np.asarray([0.977, 0.891, 0.862, 0.811, 0.809, 0.775, 0.724])


plt.figure()

plt.title("GENKI Smile Accuracy against Distortion (MSE)")
plt.gca().set_xlim([0.00, 0.002])
plt.gca().set_ylim([0.5, 1])
plt.xlabel("Distortion: MSE per pixel")
plt.ylabel("Smile Classification Accuracy")
plt.plot(mse_loss, laplace_smile, "ro--")
plt.plot(mse_loss, uniform_smile, "bx--")
plt.plot(mse_loss, GAP_smile, "yD-")
plt.legend(["Adding Independent Laplace Noise", "Adding Independent Uniform Noise", "Generative Adversarial Pivacy"])
plt.show()
# plt.savefig("gender_acc.jpg")
plt.close()

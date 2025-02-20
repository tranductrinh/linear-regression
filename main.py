import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load training dataset
data = pd.read_csv('./train.csv')

# Visualize 1000 houses
NUMBER_OF_HOUSES = 1000
plt.plot(data['GrLivArea'][:NUMBER_OF_HOUSES], data['SalePrice'][:NUMBER_OF_HOUSES], 'ro')
plt.title('Sale Price vs Living Area (automatically closed in few seconds)')
plt.xlabel('Ground Living Area')
plt.ylabel('Sale Price')
plt.show(block=False)
plt.pause(5)  # Wait for few seconds
plt.close()

# Gradient descent constants
LEARNING_RATE = 0.01
MAX_NUMBER_OF_ITERATIONS = 1000

gr_liv_area = data['GrLivArea']
sale_price = data['SalePrice']
m = sale_price.size

# Pick some random theta values to start with
np.random.seed(123)
initial_theta = np.random.rand(2)

# Standardization and add a bias term
gr_liv_area = (gr_liv_area - gr_liv_area.mean()) / gr_liv_area.std()
gr_liv_area = np.c_[np.ones(gr_liv_area.shape[0]), gr_liv_area]

def gradient_descent(gr_liv_area, sale_price, theta, iterations, learning_rate):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        sale_price_predict = np.dot(gr_liv_area, theta)
        cost = sale_price_predict - sale_price
        total_cost = 1 / (2 * m) * np.dot(cost.T, cost)
        past_costs.append(total_cost)
        theta = theta - (learning_rate * (1 / m) * np.dot(gr_liv_area.T, cost))
        past_thetas.append(theta)

    return past_thetas, past_costs


past_thetas, past_costs = gradient_descent(gr_liv_area, sale_price, initial_theta, MAX_NUMBER_OF_ITERATIONS, LEARNING_RATE)
optimal_theta = past_thetas[-1]

print("Theta 0 and theta 1: {:.2f}, {:.2f}".format(optimal_theta[0], optimal_theta[1]))

# Visualize the cost function
plt.title('Cost Function (automatically closed in few seconds)')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show(block=False)
plt.pause(5)  # Wait for few seconds
plt.close()

##### Animation

# Set the plot up
fig = plt.figure()
ax = plt.axes()
plt.title('Sale Price vs Living Area')
plt.xlabel('Ground Living Area')
plt.ylabel('Sale Price')
plt.scatter(gr_liv_area[:, 1], sale_price, color='red')
line, = ax.plot([], [], lw=2)
annotation = ax.text(-1, 700000, '')
annotation.set_animated(True)
plt.close()

# Generate the animation data
def init():
    line.set_data([], [])
    annotation.set_text('')
    return line, annotation


# Animation function
def animate(i):
    x = np.linspace(-5, 20, 1000)
    y = past_thetas[i][1] * x + past_thetas[i][0]
    line.set_data(x, y)
    annotation.set_text('Cost = %.2f e10' % (past_costs[i] / 10000000000))
    return line, annotation

print("Generating animation...")
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=0, blit=True)
anim.save('animation.gif', writer='pillow', fps=30)
print("Done!")

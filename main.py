import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('./train.csv')

# Plot data
NUMBER_OF_HOUSES = 1000
plt.plot(data['GrLivArea'][:NUMBER_OF_HOUSES], data['SalePrice'][:NUMBER_OF_HOUSES], 'ro')
plt.title('Sale Price vs Living Area')
plt.xlabel('Ground Living Area')
plt.ylabel('Sale Price')
plt.show()

# Gradient descent
LEARNING_RATE = 0.01
MAX_NUMBER_OF_ITERATIONS = 2000
GR_LIV_AREA = data['GrLivArea']
SALE_PRICE = data['SalePrice']
n = SALE_PRICE.size
np.random.seed(123)
# Pick some random THETA values to start with
THETA = np.random.rand(2)

# Standardization and add a bias term
GR_LIV_AREA = (GR_LIV_AREA - GR_LIV_AREA.mean()) / GR_LIV_AREA.std()
GR_LIV_AREA = np.c_[np.ones(GR_LIV_AREA.shape[0]), GR_LIV_AREA]


def gradient_descent(gr_liv_area, sale_price, theta, iterations, learning_rate):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        sale_price_predict = np.dot(gr_liv_area, theta)
        error = sale_price_predict - sale_price
        cost = 1 / (2 * n) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (learning_rate * (1 / n) * np.dot(gr_liv_area.T, error))
        past_thetas.append(theta)

    return past_thetas, past_costs


past_thetas, past_costs = gradient_descent(GR_LIV_AREA, SALE_PRICE, THETA, MAX_NUMBER_OF_ITERATIONS, LEARNING_RATE)
THETA = past_thetas[-1]

print("Gradient Descent: {:.2f}, {:.2f}".format(THETA[0], THETA[1]))

# Plot the cost function
plt.title('Cost Function')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()

##### Animation

# Set the plot up
fig = plt.figure()
ax = plt.axes()
plt.title('Sale Price vs Living Area')
plt.xlabel('Ground Living Area')
plt.ylabel('Sale Price')
plt.scatter(GR_LIV_AREA[:, 1], SALE_PRICE, color='red')
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


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=0, blit=True)
anim.save('animation.gif', writer='pillow', fps=30)

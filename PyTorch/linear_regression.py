# This is a sample Python script.
import torch
import numpy
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def compute_error_for_line_given_points(b,w,points):
    totalError=0
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError+=(y-(w*x+b))**2
    return totalError/float(len(points))

# every step the update of variable
def step_gradient(b_current,w_current,points,learningRate):
    b_gradient=0
    w_gradient=0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error=(w_current*x)+b_current-y
        b_gradient+=(2/N)*error
        w_gradient+=(2/N)*error*x
    return [b_current-learningRate*b_gradient,w_current-learningRate*w_gradient]

def gradient_descent_runner(points,starting_b,starting_w,learning_rate,num_iterations):
    b=starting_b
    w=starting_w
    for i in range(num_iterations):
        b,w=step_gradient(starting_b,starting_w,points,learning_rate)
    return [b,w]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print(torch.__version__)
    print(torch.cuda.is_available())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

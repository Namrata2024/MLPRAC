


import matplotlib.pyplot as plt


x=[1,2,3,4,5]
y=[2,4,5,4,5]


x_mean=sum(x)/len(x)
y_mean=sum(y)/len(y)


numerator=sum((xi-x_mean)*(yi-y_mean) for xi,yi in zip(x,y))
denominator=sum((xi-x_mean)**2 for xi in x)


m=numerator/denominator

b=y_mean-m*x_mean

def predict(x):
    return m*x+b


new_x=int(input("Enter the value of x for testing the regression model : "))

print(f'predicted value of y is: {predict(new_x)}')

plt.scatter(x,y,label='data points')

regression_line=[predict(val)for val in x]

plt.plot(x,regression_line,color='red',label='regression line')

plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()


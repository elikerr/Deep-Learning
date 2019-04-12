def third_degree_polynomial(x, constants_vector):
    #This function takes an input value x and a vector of coefficients and returns the value 
    # of a third degree polynomial, evaluated at that point, with those coefficients.
    y=[]
    for i in x:
        sum=0
        for j in range(len(constants_vector)):
            sum=sum+((i**j)*constants_vector[j])
        y.append(sum)
    return y

def fit(degree, inputs, targets):
    #This function takes the degree of a polynomial and a set of [x,y] data 
    # and returns the coefficients of the best fir polynomial (of the specified degree)
    matrix=[]
    #runs through each input value x_i
    for i in inputs:
        row=[]
        #calculate each row element using given input values and add it to the row
        for j in range(degree+1):
            rowelement=float(i**(j))
            row.append(rowelement)
        #add each row to the design matrix
        matrix.append(row)
    #convert matrix from list to array
    X=np.array(matrix)        
    
    return np.linalg.inv(X.T@X)@X.T@targets

def any_degree_polynomial(x, constants_vector):
    ##This function takes a vector of input values (length [batch,1]) as well as a vector of 
    #coefficients(length [degree,1]). It will return the vector of output values (length [batch,1])
    # corresponding to given input vector passed through an nth order polynomial with the given coefficients.
    y=[]
    for i in x:
        sum=0
        for j in range(len(constants_vector)):
            sum=sum+((i**j)*constants_vector[j])
        y.append(sum)
    return y

#Fit a 3rd degree polynomial to data and visualize the result of your fit
#
#plot the data that we generated previously
plt.plot(x_ex1_train, y_ex1_train, label = 'data');
plt.title('Real Data vs Estimator Model');
plt.xlabel('Input');
plt.ylabel('Ground truth');
#
#calculate and plot best fit polynomial
coefficients = fit(3,x_ex1_train, y_ex1_train)
y_hat=any_degree_polynomial(x_ex1_train, coefficients)
plt.plot(x_ex1_train, y_hat, 'r',label='estimator')
plt.legend()

def mse(predicted_values, targets):
    ##Takes two vectors of equal length and calculates the mean-squared error.
    mse=sum((predicted_values-targets)**2)/(len(targets))
    return mse

##This section of code was intended to iteratively generate a best fit for our given data
# using polynomials of varying degree. It then compared each model to a validation set.
# The purpose of the exercise was to demonstrate the effects of overfitting to your training data.
sqerrortrain=[]
sqerrorvalidation=[]
for i in range(10):
    coeffs=fit(i+1,x_ex1_train,y_ex1_train)
    modelt=any_degree_polynomial(x_ex1_train,coeffs)
    modelv=any_degree_polynomial(x_ex1_val,coeffs)
    sqerrort=mse(modelt,y_ex1_train)
    sqerrorv=mse(modelv,y_ex1_val)
    sqerrortrain.append(sqerrort)
    sqerrorvalidation.append(sqerrorv)

plt.plot(sqerrortrain, label = 'training');
plt.title('MSE for Training and Validation Sets');
plt.xlabel('Degree-1');
plt.ylabel('MSE');
plt.plot(sqerrorvalidation, 'r',label='validation')
plt.legend()



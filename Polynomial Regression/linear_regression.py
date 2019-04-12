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
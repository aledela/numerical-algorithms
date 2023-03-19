# Angela Ale De La Cruz
# Implementation of various numerical analysis algorithms using Python.

import numpy as np
from numpy import *

def bisection(f,a,b,tol=1.0e-6):
    fa = f(a)
    fb = f(b)
    roots = []
    if np.sign(fa)*np.sign(fb) >= 0:
        print 'f(a)f(b)<0 not satisfied!'
        quit()
    while (b-a)/2. > tol:
        c = (a+b)/2
        roots.append(c)
        fc = f(c)
        if fc == 0:
            return c
        if np.sign(fc)*np.sign(fa) < 0:	# a and c make the new interval
            b = c
            fb = fc
        else:								# c and b make the new interval
            a = c
            fa = fc
        roots.append(c)
    return (a+b)/2, np.array(roots)					# new midpoint is best estimate

def polynest(x,coeff,b=[]):
	d = len(coeff)-1
	if b ==[]:
		b = zeros(d)
	y = coeff[d]
	for i in range(d-1,-1,-1):
		y *= (x-b[i])
		y += coeff[i]
	return y

def rowswap(Mat,j,k):
    Mat[[j,k],:]=Mat[[k,j],:]
    return Mat

def rowdiff(Mat,j,k,scale=1.0):
    Mat[j,:] = Mat[j,:]-np.float(scale)*Mat[k,:]
    return Mat

def rowscale(Mat,j,scale=1.0):
    Mat[j,:] = scale*Mat[j,:]
    return Mat

#Factorization of Matrix Mat into L,U,P.
def lu(Mat):
    A=np.asarray(Mat)
    (m,n)=A.shape
    if (m!=n):
        raise ValueError("The input matrix must be square.")
    L=np.zeros(A.shape)
    P=np.identity(n)

    for col in range(n):
        pivot_index=np.argmax(np.absolute(A[col:n,col]))
        pivot_index+=col
        pivot =np.float(A[pivot_index,col])
        if(pivot):
            if (pivot_index!=col):
                A=rowswap(A,pivot_index,col)
                L=rowswap(L,pivot_index,col)
                P=rowswap(P,pivot_index,col)
        else:
            raise ValueError("The matrix provided is singular.The decomposition failed.")
        for row in range(col+1,n):
            scalefactor=A[row,col]/pivot
            L[row,col]=scalefactor
            A=rowdiff(A,row,col,scalefactor)

    L+=np.identity(n)
    return L,A,P

def backsub (U,b):
    U=np.asarray(U)
    b=np.asarray(b)
    n=b.size
    (mU,nU)=U.shape
    if (mU!=nU) or (n!=nU):
        raise ValueError("The dimension of the input are not correct.")

    x=np.zeros(b.shape)
    x[n-1]=b[n-1]/np.float(U[n-1,n-1])
    for row in range(n-2,-1,-1):
        x[row]=(b[row]-np.dot(U[row,row+1:n],x[row+1:n]))/U[row,row]

    return x

def forwardsub(L,b):
    L=np.asarray(L)
    b=np.asarray(b)
    n=b.size
    (mL,nL)=L.shape
    if (mL!=nL) or (n!=nL):
        raise ValLeError("The dimension of the inpLt are not correct.")

    x=np.zeros(b.shape)
    x[0]=b[0]/np.float(L[0,0])
    for row in range(1,n):
        x[row]=(b[row]-np.dot(L[row,0:row],x[0:row]))/L[row,row]

    return x

#Conducts a forward backward solve of the system LUx=b
def fbsolve(L,U,b):
    L=np.asarray(L)
    U=np.asarray(U)
    b=np.asarray(b)
    n=b.size
    (mU,nU)=U.shape
    (mL,nL)=L.shape
    if (mL!=nL) or (n!=nL) or (mU!=nU) or (nU!=n):
        raise ValueError ("The dimension of the input are not correct.")
    y=forwardsub(L,b)
    x=backsub(U,y)
    return x

#Solves the equation Ax=b using forward backward substitution.
def lusolve(A,b):
    L,U,P=lu(A)
    x=fbsolve(L,U,np.dot(P,b))
    return x


def fixedpt(gFun,xInit,tol=1.0e-6, maxIter = 4000):
    k=0  # iterating factor 0 to n for the roots
    rk = xInit # double r_0 to r_k, initialized as r_0
    rkCurrent = 0 # double for current r_(k+1), initialized as 0
    roots = [] # list of roots
    err = abs(rkCurrent-rk) # double, current error = |r_(k+1) - r_k|

# Iterate
    while err>tol and k<maxIter:
        roots.append(rk)  #add r_k to the list of roots
        rkCurrent = gFun(rk) #r_k+1 = g(r_k)
        err = abs(rkCurrent-rk) #recalculating error every iteration
        rk = rkCurrent # updating r_k as r_(k-1)
        k = k + 1 # increading k by 1 each iteration

    return rk, np.asarray(roots) #returns r_k and array of roots

def newton(fxn,dfxn,xInit,tol=1.0e-6):
    k=0  # iterating factor 0 to n for the roots
    rk = xInit # double r_0 to r_k, initialized as r_0
    rkCurrent = 0 # double for current r_(k+1), initialized as 0
    roots = [] # list of roots
    err = abs(rkCurrent-rk) # double, current error = |r_(k+1) - r_k|

# Iterate
    while err>tol:
        roots.append(rk)  #add r_k to the list of roots
        rkCurrent = rk-(fxn(rk)/dfxn(rk)) #r_k+1 = g(r_k)
        err = abs(rkCurrent-rk) #recalculating error every iteration
        rk = rkCurrent # updating r_k as r_(k-1)
        k = k + 1 # increading k by 1 each iteration


    return rk, np.asarray(roots) #returns r_k and array of roots

def jacobi(A,b,xinit,tol=1e-6,maxIter = 4000):
    k=0  # iterating factor 0 to n for the roots
    xk = xinit # double r_0 to r_k, initialized as r_0
    xkCurrent = A # double for current r_(k+1), initialized as 0
    err = 1

# Iterate
    while (err>tol) and k<maxIter:
        xkCurrent = (np.linalg.inv(np.diag(np.diag(A)))).dot(b-((np.tril(A,-1)+ np.triu(A,1)).dot(xk)))
        err = abs(xkCurrent[0]-xk[0]) #recalculating error every iteration
        xk = xkCurrent # updating r_k as r_(k-1)
        k = k + 1
    return xk
def gausssiedel(A,b,xinit,tol=1e-6,maxIter = 4000):
	A=np.array(A)
	x = xinit
	it_count=1
	while it_count<=maxIter:
	    x_new = np.zeros_like(x)
	    print("Iteration {0}: {1}".format(it_count, x))
	    for i in range(A.shape[0]):
	        s1 = np.dot(A[i, :i], x_new[:i])
	        s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
	        x_new[i] = (b[i] - s1 - s2) / A[i, i]
	    if np.allclose(x, x_new, rtol=1e-8):
	        break
	    x = x_new
        it_count=it_count+1
	return x



def leastSquares_lu(A,b):
    AtA = np.dot(np.transpose(A),A)  # If we didn't want to store At we could just use np.dot(np.transpose(A),A)
    y = np.dot(np.transpose(A),b)
    x = lusolve(AtA,y)

    return x, b - np.dot(A,x)


    # Input: a vector x
    # Output the householder reflector I-2P mapping x to ||x||*(1,0,0,...0) of the same length
def householder(vec):
	w = np.zeros(vec.size)
	w[0] = linalg.norm(vec, 2)
	v = (w - vec).reshape(vec.size,1)
	print(w)
	b = np.dot(np.transpose(v), v)
	a= np.ones(b.shape)
	P = np.dot(v, np.transpose(v))* np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	print("P:")
	print(P)
	H = np.identity(P.shape[0]) - 2*P
	return H



# Solves QR using Gram Schmidt
def qr_gramschmidt(A):
	Q = np.zeros(A.size).reshape(A.shape)
	R = np.zeros(A.shape[1] ** 2).reshape(A.shape[1], A.shape[1])
	for j in range(A.shape[1]):
		y = A[:,j]
		for i in range(j):
			R[i][j] = np.dot(Q[:,i], A[:,j])
			y = y - R[i][j] * Q[:,i]
		R[j][j] = linalg.norm(y, 2)
		Q[:,j] = y / R[j][j]
	return Q, R


    # Input: a matrix Mat
    # Output: an orthogonal matrix Q and an upper triangular matrix R so that QR=Mat
    # This implementation uses Householder reflectors.
def qr(A):
	vec = A[:,0]
	h1=householder(vec)
	h=h1
	R = h.dot(A)
	row,col=A.shape
	Q=h
	for i in range(1,row):
		for j in range(1,col):
			h = householder(np.transpose(A[i:row,j:col]))
			h_adjusted=np.identity(len(h1))
			for n in range(0,len(h)):
				for m in range(0,len(h)):
					h_adjusted[n+1,m+1]=h[n,m]
			R = h_adjusted.dot(R)
			Q = h_adjusted.dot(Q)
	return Q,R


    # Input: (mxn) matrix A and n vector b
    # Output: x so that Ax=b if A is square or x is least squares solution if m>n
    #                resid as the error of the least squares
def qrsolve(Mat,vec):
    m,n=Mat.shape
    if m==n:
        x,res=lusolve(Mat,vec),0
    else:
        x,res = leastSquares_lu(Mat,vec)
	return x,res


def newtondd(x,y):
    n = len(x)
    v = zeros((n,n))
    for j in range(n):
        v[j,0] = y[j]
    for i in range(1,n):
        for j in range(n-i):
            v[j,i] = (v[j+1,i-1]-v[j,i-1])/(x[j+i]-x[j])
    c = v[0,:].copy()
    return c

def newtonInterp(x,y):
    c=newtondd(x,y)
    return polynest(x,c)


def chebyshevRoots(numrts):
    rts=[]
    for n in range(numrts):
        rts.append(np.cos((2*n + 1)*np.pi/(2*(numrts))))
    return rts

def chebyshevRootsInterval(n, interv):
    roots = np.zeros(n)
    a = interv[0]
    b = interv[1]
    for i in range(0, n):
        x_i = np.cos((2*i + 1)*np.pi/(2*(n)))*((b-a)/2) + ((b + a)/2)
        roots[i] = x_i
    return roots
def chebyshevInterp(func,n,x):
    roots=chebyshevRootsInterval(x)
    if n == 0:
        y = 1
    elif n == 1:
        y = func
    else:
        y = 2func*chebyshevInterp(func,n-1,x) - chebyshevInterp(func,n-2,x)
    return y

#Cubic Splines method as shown in the book
def cubiccoeff(x,y,option=1,v1=0,vn=0):
	n = len(x);
	A = zeros((n,n))
	r = zeros(n)
	dx = x[1:] - x[:-1]
	dy = y[1:] - y[:-1]
	for i in range(1,n-1):
		A[i,i-1:i+2] = ( (dx[i-1], 2*(dx[i-1]+dx[i]), dx[i]) )
		r[i] = 3*(dy[i]/dx[i] - dy[i-1]/dx[i-1])
	if   option==1:
		A[ 0, 0]  =  1.
		A[-1,-1]  =  1.
	elif option==2:
		A[ 0, 0] = 2; r[ 0] = v1
		A[-1,-1] = 2; r[-1] = vn
	elif option==3:
		A[ 0, :2] = [2*dx[  0],  dx[  0]]; r[ 0] = 3*(dy[  0]/dx[  0]-v1)
		A[-1,-2:] = [  dx[n-2],2*dx[n-2]]; r[-1] = 3*(vn-dy[n-2]/dx[n-2])
	elif option==4:
		A[ 0, :2] = [1,-1]
		A[-1,-2:] = [1,-1]
	elif option==5:
		A[ 0, :3] = [dx[  1], -(dx[  0]+dx[  1]), dx[  0]]
		A[-1,-3:] = [dx[n-2], -(dx[n-3]+dx[n-2]), dx[n-3]]
	coeff = zeros((n,3))
	coeff[:,1] = linalg.solve(A,r)
	for i in range(n-1):
		coeff[i,2] = (coeff[i+1,1]-coeff[i,1])/(3.*dx[i])
		coeff[i,0] = dy[i]/dx[i]-dx[i]*(2.*coeff[i,1]+coeff[i+1,1])/3.
	coeff = coeff[0:n-1,:]
	return coeff

def simpson(func,x0,x2):
    x1=x0+(x2-x0)/2.0
    h=x2-x1
    return (h/3)*(func(x0)+4*func(x1)+func(x2))

def compositeSimpson(func,m,a,b):
    h=(b-a)/(2.0*m)
    interval = np.linspace(a,b,2*m)
    sum1=0
    sum2=0
    for i in range(m):
        sum1+=func(interval[2*i])
    for j in range(1,m):
        sum2+=func(interval[2*j-1])
    return (h/3)*(func(interval[0])+func(interval[2*m-1])+4*sum1+2*sum2)

def adaptiveSimpson(func,a,b,tol=0.000001):
    c = 0.5*(a+b)
    i1 = simpson(func,a,b)
    i2 = simpson(func,a,c) + simpson(func,c,b)
    err = np.abs(i2 - i1)
    if err < tol:
        return i2
    else:
        tol *= 0.5
        return adaptiveSimpson(func,a,c,tol) + adaptiveSimpson(func,c,b,tol)

# UNCOMPLETED

# def gaussQuad(func,m,a,b):
#     return 0

# Needs editing to make sure results are accurate in all cases.
def singularVD(A,k=1):
    evalue1,evector1 = np.linalg.eig(A.dot(A.T))
    evalue2,evector2 = np.linalg.eig(A.T.dot(A))


    # Sort eigenvectors and corresponding eigenvalues in decending order
    idx1 = evalue1.argsort()[::-1]
    evalue1 = evalue1[idx1]
    evector1 = evector1[:,idx1]
    idx2 = evalue2.argsort()[::-1]
    evalue2 = evalue2[idx2]
    evector2 = evector2[:,idx2]

    # Create S,V,D matrix
    U = evector1
    temp = np.diag(np.sqrt(absolute(evalue2)))
    S = np.zeros_like(A).astype(np.float64)
    S[:temp.shape[0],:temp.shape[1]] = temp
    V = evector2.T

    # Initialize
    sum=0
    svdi=0
    nU,mU=U.shape
    nV,mV=V.shape
    for i in range(k):
        si = S[i,i]
        ui = U[:,i].reshape(mU,1)
        vi=V[i,:].reshape(1,mV)
        svdi=dot(ui*si,vi)
        sum+=svdi
    # Uncomment to see the resulting image for $p=k$ terms
    # plt.imshow(sum, cmap=plt.get_cmap("gray"))
    # plt.show()
    return U,S,V,sum

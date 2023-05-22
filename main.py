import numpy as np

'''BASIC CODING'''
## ONE DIMENSIONAL LIST
a= np.array([1, 3, 4])
#print(a)
d= np.array([1, 3, 4], dtype='int16')
#print(d)
## TWO DIMENSIONAL LIST

b= np.array([[1, 9, 2, 8], [3, 7, 4, 6]])
#print(b)

## THREE DIMENSIONAL LIST
c= np.array([[1.0, 3.0, 4.0, 5.0, ],  [2.0, 3.0, 4.0, 5.0 ], [3.0, 5.0, 9.0, 6.0]])
#print(c)

'''GETTING DIMENSION OF THE ARRAYS ==> ndim FUNCTION'''
#To get the dimesnion of NumPy arrays, we will be using the ndim function effectively.
#The ndim allows us to get to know abot the dimension of each of arrays.
#print(a.ndim)
#print(b.ndim)
#print(c.ndim)

'''GETTING THE SHAPE OF ARRAYS ==> shape FUNCTION'''
#To get the Shape of NumPy arrays, we will be using the ndim function effectively.
#The shape allows us to get to know abot the dimension of each of arrays.
#print(a.shape)
#print(b.shape)
#print(c.shape)

'''GETTNG THE TYPE OF ARRAY ==> dtype FUNCTION'''
#print(a.dtype) # ==> Here, ususlaly it has int32 & takes higher bit, we can specify it as int16
#print(b.dtype)
#print(c.dtype)

##   SPECIFYING LOWER BITS.
#d= np.array([1, 3, 4], dtype='int16')
#print(d)
#print(d.ndim)
#print(d.shape)
#print(d.dtype)

'''GETTING THE SIZE OF THE ARRAY'''
#print(a.size)
#print(b.size)
#print(c.size)
#print(d.size)

'''GETTING THE ITEM SIZE OF THE ARRAY'''
#print(a.itemsize)
#print(b.itemsize)
#print(c.itemsize)
#print(d.itemsize) #it will have lower size as it specify int16 as compare to a.

'''NUMBER OF BYTES USED TO STORE THE ARRAY DATA'''
#print(a.nbytes)
#print(d.nbytes)
#print(b.nbytes)
#print(c.nbytes)

'''ASSESSING/CHANGING SPECIFIC ELEMENTS, ROWS, COLUMNS ETC. '''

'''A==> Getting Specific Elements'''
# Create any Array
e= np.array([[1, 6, 3, 8, 3, 9, 5], [4, 5, 8, 2, 1, 5, 3]])
#print(e)
#print(e.shape)
# Now you can recieve or get any specifc element in arry using [row, columns] indexing.
#print(e[0, 5]) #=> It will give you the answer of 9
#print(e[1, 5]) #=> It will give you the answer of 5

'''GETTING A SPECIFIC ROW'''

# so, We will be using same indexing (Rows, column).
# In order to ger specific row, we need just index it in square bracket.==> e[0
# Now, that we don't want any column, we can use empty columns string which ==> :
# So, it can be e[0, :]
#print(e[0, :])

'''GETTING A SPECIFIC COLUMNS'''
## like, row function, we will be using same indexing [rows, column].
# Now that, we dont wany row, we can put empty function for row as ==> :
# so, it will be [:, index for column=> 2] = [:, 2].
#print(e[:, 2])

'''ADVANCED ROW'S SPECIFC FUNCTIONS'''
# In this function we will be using some advanced Function.
# Here, we want to get values in the specific rows but differently.
# Here, we received the values in row but with intervals.
# The interval values means that, one values than next values will be with interval.
# The code is simple, it will be using same indexing [rows, columns].
# The code start from [row index, starting index, ending index, interval]
#print(e[0, 1: 6: 2])

'''CHANGING THE SPECIFIC ELEMENT'''
# The NumPy specifically helps us change the or replace the values of specific element in array.
# The following is the code
#e[1, 2] = 457575  # create varaible name with specific elements and number that will replace it
#print(e) #Executing Function.

'''CHANGING THE VALUE OF SPECIFIC COULMN'''
#e[:, 2] = 3938393
#print(e)

# Following Code Can Also be Used.
#e[:, 2] = [4, 34]
#print(e)

'''WORKING WITH 3D ARRAYS'''
# To code 3D arrays, we need to have values inside of double SQUARE bracket
# DON'T FORGET DOUBLE BRACKET IN END OF SECOND AND START OF THRID ROW.
f = np.array([[[34, 76], [74, 64]] , [[97, 67], [18, 92]]])
#print(f.ndim) #  => The 3 numbers shows that, its 3D array

'''GETTING SPECIFIC ELEMENT INSIDE 3D ARRAY'''
#print(f)
#print(f[0, 1, 1]) #=> It will give us answer of 64
#       UNDERSTANDING INDEXING PATTERN
# In 3D, 4 brackets and grouped into two on lines of 3d Shape
# First Two brackets are grouped as first index and last two barcker are grouped into second index.
# Now, you will have to two rows in first indexed 2 brackets. So, Specify its index
# In last, now you specify the idex.
# CODE== [Indexed brackets, rows index, elements index] ==> print(f[0, 1, 1])

'''SOME FANCY OPERATION IN THE 3D ARRAYS'''

## GETTING THE SECOND ROWS OF EACH SIDE
#print(f[:, 1, :])

## REPLACING VALUES TO BE STORED IN VARAIBLE.
#f[:, 1, :] = [[9, 8] , [2, 4]]
#print(f)


'''INITIALIZING DIFFERENT TYPES OF ARRAYS'''

g= np.zeros(5)
#print(g) #It will print row of 5 zeros

h= np.zeros((2 ,4))
#print(h)
#It will print 2 rows with four column of Zeros.

i= np.zeros((3, 2, 4))
#print(i)
#three blocks of 2 rows and 4 columns of Zeros
'''NOTE: We need to add Double Bracket once we're initializing more than one rows'''

'''INITIALIZING OTHER NUMBERS ===>   np.full  ==>   '''
# We can also initialize the other numbers apart from the zeros.
# Here we will be using the np.full function

#j= np.full((2,2), 100)
#print(j)

# 2 Rows, 2 Columns of 100

#ja= np.full((2, 4, 3), 100)
#print(ja)
# The Above codes will print 2 blocks each contain 4 rows and 3 columns

'''FULL LIKE OR LIKE'''
# The following code will imitate shape of already developed array.
# And it will fill it with the number which we will be providing.

#           FULL LIKE FUNCTION
k = np.full_like(e, 4)
#print(k)
#    The Full Like Function will take two arguments
        # 1. Varaible name of array
        # 2. number which we want to fill it with.

#           SHAPE FUNCTION
ka = np.full(e.shape, 5)
#print(ka)
# The Shape function will utilize the full argument along with shape argmument inside bracket.
# After that, it will ask number

'''THE RANDOM DECIMAL INITIATION FUNCTION'''

'''             RAND FUNCTION  '''
#la = np.random.rand(2, 3)
#print(la)
#So, Above code will be create 2 row 3 column decimal of random numbers not a specific
# It will create random not decimals.

#lb= np.random.rand(2, 4, 3)
#print(lb)
#Above Code will create the following
# 2 Blocks
# 4 rows in each blocks
# 3 columns in each block

'''RANDOM SAMPLE FUNCTION'''
#lc= np.random.random_sample(e.shape)
#print(lc)
#The Above code will take the shape of e and create random numbers of decimals.

'''RANDOM INTEGERS VALUES  ==> random.ranint'''
# The python also allows us to create the matrix of RANDOM INTEGERS VALUES.

o = np.random.randint(2, 7, size=(2, 4, 5))
#print(o)
'''BREAKDOWN OF CODE'''
# The Random function has its sub-fucntion of randint.
# So, it will take three parameters
# 1. Starting index= The index which is minimum.
# 2. Ending index = The index which is maximum
# 3. Size paramters = You can say shape=> No. of blocks, rows and columns.

'''IDENTITY MATRIX'''  #Learn more about the identity Matrix

#p= np.identity(5)
#print(p)

'''REPEATING AN ARRAY'''
#We can repart the arrays both one dimension and two dimensional.

# Repeating array one dimensional.

array1A = np.array([[2, 3, 4]])
rrrr= array1A.repeat(3, axis=1)
#print(rrrr)


array1 = np.array([[1, 2, 3]])
rrr = array1.repeat( 3, axis=0)
#print(rrr)

## Both the code are same.
# For repeating uni-dimensioanl array, keep axis =1
# For repeating 2D or 3D dimeniosnal keep axis =0

'''LETS HAVE THE MATRIX GAME'''
# The matrix game is all about the playing with matrices.
# In the matrix game, we will be creating two different matrix
# These two matrices will be different and seprate.
# We than combine these two matrix into 3rd.

## 1. FIRST MATRIX
#     Create 5, 5 matrix of ones only

first = np.ones((5, 5))
#print(first)

## 2. SECOND MATRIC
# Create second 3, 3 matric of zeros only
second = np.zeros((3, 3))
#print(second)

## Now Change middle 0 into 9

second[1, 1] = 9
#print(second)

# Now Incorporate entire seond matrix into first

first [1:4, 1:4] = second
#print(first)

'''ALWAYS BE CAREFUL IN THE COPYING ARAYS'''
#The copying arrays function can yield into some sort of errors.
# Basically, we copy one array into the another arrays
# Like, we have array A and we want to copy values of array A into Array B.
# But, if we change some value into B, It will affect A. Like See following

#a = np.array([1, 2, 3]) #We created array a
#b = a #We coped values of array a into array b
#print(b)
#b[1] = 90 # Now we changed values of second index into 90
#print(b) # Now we have updated
# Now execute A
#print(a)
'''you will not see the original values of a which is 1, 2, 3 but 1, 90, 3'''
'''HOW TO BE CAREFUL IN COPYING===> by using copying function = copy.()'''

#aa = np.array([10, 20, 30, 40])
#print(aa)
#bb = aa.copy()
'''above is hack'''
#print(bb)
#bb[2] = 3000
#print(bb)
#print(aa)


'''                       MATHAMATICS IN NUMPY                                     '''

'''1. BASIC ARITHMATIC FUNCTIONS'''
ma = np.array([10, 20, 30, 40])

# Addition
#maa = ma + 2
#print(maa)

#Substraction
#maa1 = ma - 2
#print(maa1)

#Multiplication
#maa2 = ma * 2
#print(maa2)

#Division
#maa3 = ma / 2
#print(maa3)

#Exponential Power

#maa4 = ma ** 2
#print(maa4)

''' 2. ADDING and OTHERS TWO ARRAYS MAY BE'''

# 1. Addition
na = np.array([15, 25, 35, 45])
na1 = ma + na
#print(na1)

# 2. Substraction
na2 = ma - na
#print(na2)

#3. Multiplications
na2 = ma * na
#print(na2)

#Division
na3 = ma / na
#print(na3)

'''3. SOME TRIGNOMETRY '''

# Sin theta
ja = np.sin(ma)
#print(ja)
# cos theta
ja1 = np.cos(ma)
#print(ja1)

'''BASIC LINEAR ALGERBRA'''

##    Multiplication of two matrixes

ones = np.ones((2, 3))
#print(ones)

twos = np.full((3, 2), 2)  #by rules, we need to have second matrix with 3 rows and 2 columsn
                           # If we have first matrix with 2 rows and 3 columns
#print(twos)

threes = np.matmul(ones, twos)
#print(threes)

'''         FINDING DETERMINANTS    '''

det = np.identity(3)
#print(det)
iden = np.linalg.det(det)
#print(iden)

'''                         SOME STATISTICS WITH NumPY              '''

brand_equity = np.array([[100, 200, 300], [40, 500, 600]])

price = np.array([100, 250, 80, 1300])
brands= np.array([13, 17, 26, 10])
#print(brand_equity)

'''SOME STATISTICAL OPERATIONS'''
#print(np.min(brand_equity))
#print(np.max(brand_equity))
#print(np.min(brand_equity, axis=1))
#print(np.max(brand_equity, axis=1))
#print(np.sum(brand_equity))
#print(np.sum(brand_equity, axis=1))
#print(np.mean(brand_equity))
#print(np.mean(brand_equity, axis=1))
#print(np.corrcoef(brand_equity)) => Correlation Co-efficient
#print(np.corrcoef(price, brands)) => Correlation Co-efficient
'''
        SOME OTHER STATISTICAL OPERATIONS 
1. Mean: np.mean().
2. Median: np.median().
3. Standard deviation: np.std().
4. Variance:  np.var().
5. Product:  np.prod().
6. Percentile: np.percentile().
7. Correlation coefficient: np.corrcoef().
8. Covariance matrix:  np.cov().
9. Histogram:  np.histogram() to visualize the distribution of data.'''

# print(np.std(price))
# print(np.percentile(price, brands))
# print(np.var(price))
# print(np.prod(brands))
# print(np.cov(price))


'''                              RE-ARRANGING THE ARRAYS          '''
'''1. RE-SHAPPING THE ARRAYS'''

data = np.array([[10, 20, 30, 70], [40, 50, 60, 80]])
#print(data)

reshaped_data = data.reshape((8, 1))
# print(reshaped_data)

reshaped_data1= data.reshape((2, 2, 2))
# print(reshaped_data1)

reshaped_data2= data.reshape((4, 2))
# print(reshaped_data2)

'''VERTICAL STACKS'''
# vertical stacking IS process of combining 2 or more arrays vertically along the row axis (axis 0).
# It concatenates the arrays vertically to create a new array with a greater number of rows
v1 = np.array([1300, 1400, 1500])
v2= np.array([1600, 2100, 1627])
# print(v1)
# print(v2)
v3= np.vstack((v1, v2))
# print(v3)


v4= np.array([[10, 20, 30 ],
              [40, 50, 60]])

v5= np.array([[100, 200, 300],
              [400, 500, 600]])
# print(v4)
# print(v5)
v6= np.stack((v4, v5))
v61= np.stack((v4, v4, v4, v5))
# print(v61)
v62= np.stack((v4, v5, v5, v5))
# print(v62)

'''HORIZANTAL STACKING'''

horizantal = np.hstack((v4, v5))
# print(horizantal)

'''LOADING AN EXTERNAL FILE'''

file = np.genfromtxt('data.txt', delimiter=',')
# print(file.astype('int32'))
csv_file = np.genfromtxt('session.csv', delimiter=',')
# print(csv_file)

'''SOME ADVANCED INDEXING OPTIONS'''
'''1. Returning Boolean Value'''
# print(file > 50)

'''2. resturning number greater than 50'''
# print(file[file>50])
'''3. Returning specifc values in Arrays'''
test= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(test[[1, 2, 8]])

'''3. ANY OF VALUES GREATER THAN 50'''
# print(np.any(file > 50, axis=0))
# print(np.any(file > 50, axis=1))
# print(np.all(file > 50, axis=0))

'''4. Passing Multiple Conditions'''
# print((file > 50) & (file < 100))
'''Condition of Not lesser than 50'''
print((~(file > 50) & (file > 100)))



























































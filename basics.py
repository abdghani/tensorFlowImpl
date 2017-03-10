import tensorflow as tf

#this would create a constant type of tensor node with type float
node1 = tf.constant(3.0,tf.float32)
#will do the same .just float type will be initialized implicitely
node2 = tf.constant(4.0)
#addition of tensors
node3 = tf.add(node1, node2)

#A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 

#getting more complex by calling another function
add_and_triple = adder_node * 3.


#variables are not like constants 
#there values can be later assigned
#they are initialized when we call tf.global_variables_initializer P.S. line 48
W = tf.Variable([.3], tf.float32)
bias = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + bias

#now in case we want to reassign the values to a variable
#we can make use of assign fn. then we have to run sess like in line  55
fixW = tf.assign(W, [-1.])
fixb = tf.assign(bias, [1.])

#loss function i.e. squared difference between calculated input and expected input
y = tf.placeholder(tf.float32)
#tf.square just like np.square
squared_deltas = tf.square(linear_model - y)
#calculate sum accross an axis .Pls read https://www.tensorflow.org/api_docs/python/tf/reduce_sum
loss = tf.reduce_sum(squared_deltas)

#A session encapsulates the control and state of the TensorFlow runtime.
sess = tf.Session()
print(sess.run([node1,node2]))
print(sess.run([node3]))
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
print(sess.run(add_and_triple, {a: [1,3], b: [2, 4]}))


init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#reassigning the values
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


#using gradient descent optimizer of tf
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(2000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
#after 100 iteration
print(sess.run([W, bias]))

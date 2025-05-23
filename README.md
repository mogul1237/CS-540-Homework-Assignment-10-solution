# CS-540-Homework-Assignment-10-solution

Download Here: [CS 540 Homework Assignment # 10 solution](https://jarviscodinghub.com/assignment/cs-540-homework-assignment-10-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

Question 1: Neural Network [100 points, evenly among parts]
We will build a neural network to perform binary classification. Each input item has two real-valued features
x = (x1, x2), and the class label y is either 0 or 1. Our neural network has a very simple structure:
ReLU A
C Sigmoid
B ReLU
x1 x2
1 1
1
w8 w9
w2
w5 w3
w6
w1 w4
w7
There is one hidden layer with two hidden units A, B, and one output layer with a single output unit C.
The input layer is fully connected to the hidden layer, and the hidden layer is fully connected to the output
layer. Each unit also has a constant bias 1 input with the corresponding weight. See figure below. Units A
are B are ReLU, namely
fA(z) = fB(z) = max(z, 0).
Unit C is a sigmoid,
fC (z) = σ(z) = 1
1 + e−z
.
Write a program Neural.java with the following command line format:
$java Neural FLAG [args]
Where the optional arguments are real valued.
1. This neural network is fully defined by the nine weights w1, . . . , w9. We will first focus on predictions
given fixed weights.
Remark: we will use both single index and double index to refer to a weight. Single index corresponds
to the figure above. Double index, on the other hand, is used to describe the algorithm and denotes
CS 540 Fall 2017
the “from → to” nodes that the edge is connecting. For example, w8 is the same as wA,C , w2 is the
same as wx1,A, and w1 is the same as w1,A where we used “1” to denote the constant bias input of
one. These should be clear from the context.
Recall in a neural network for any unit j, it first collects input from lower units:
uj =
X
i:i→j
wijvi
where vi
is the output of lower unit i. Specifically, if i is an input unit then vi = xi
; if i is the bias
then vi = 1. The unit j then passes uj through its nonlinear function fj () to produce its output vj :
vj = fj (uj ).
When FLAG=100, arg1 … arg9 are the weights w1, . . . , w9, and arg10=x1, arg11=x2. Print uA, vA, uB, vB, uC , vC
on the same line separated by space. When printing real values, show 5 digits after decimal point by
rounding (but do not round the actual variables). For example,
$java Neural 100 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 -1
0.00000 0.00000 0.30000 0.30000 0.97000 0.72512
$java Neural 100 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 -0.2 1.7
2.18000 2.18000 1.43000 1.43000 1.34000 0.79249
$java Neural 100 4 3 2 1 0 -1 -2 -3 -4 -4 1
-6.00000 0.00000 0.00000 0.00000 -2.00000 0.11920
2. Given a training item x = (x1, x2) and its label y, the error made by the neural network on the item
is defined as
E =
1
2
(vC − y)
2
.
The partial derivative with respect to the output layer variable vC is
∂E
∂vC
= vC − y.
The partial derivative with respect to the intermediate variable uC is
∂E
∂uC
=
∂E
∂vC
f
0
C (uC ).
Recall f
0
C (uC ) = σ
0
(uC ) = σ(uC )(1 − σ(uC )) = vC (1 − vC ).
When FLAG=200, arg1 … arg9 are the weights w1, . . . , w9, arg10=x1, arg11=x2, and arg12=y. Print
E,
∂E
∂vC
, and ∂E
∂uC
on the same line separated by space. For example,
$java Neural 200 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 -1 1
0.03778 -0.27488 -0.05479

$java Neural 200 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 -0.2 1.7 0
0.31402 0.79249 0.13032
$java Neural 200 4 3 2 1 0 -1 -2 -3 -4 -4 1 0
0.00710 0.11920 0.01252
3. The partial derivative with respect to hidden layer variable vj is
∂E
∂vj
=
X
k:j→k
wjk
∂E
∂uk
.
And
∂E
∂uj
=
∂E
∂vj
∂vj
∂uj
.
Recall our hidden layer units are ReLU, for which
∂vj
∂uj
=
∂ max(uj , 0)
∂uj
=

1, uj ≥ 0
0, uj < 0 Note we define the derivative to be 1 when uj = 0. When FLAG=300, arg1 ... arg9 are the weights w1, . . . , w9, arg10=x1, arg11=x2, and arg12=y. Print ∂E ∂vA , ∂E ∂uA , ∂E ∂vB , and ∂E ∂uB on the same line separated by space. $java Neural 300 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 -1 1 -0.04383 -0.04383 -0.04931 -0.04931 $java Neural 300 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 -0.2 1.7 0 0.03910 0.03910 0.02606 0.02606 $java Neural 300 4 3 2 1 0 -1 -2 -3 -4 -4 1 0 -0.03755 0.00000 -0.05006 -0.05006 4. Now we can compute the partial derivative with respect to the edge weights: ∂E ∂wij = vi ∂E ∂uj . When FLAG=400, arg1 ... arg9 are the weights w1, . . . , w9, arg10=x1, arg11=x2, and arg12=y. Print ∂E ∂w1 , . . . ∂E ∂w9 on the same line separated by space. For example, $java Neural 400 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 -1 1 -0.04383 -0.04383 0.04383 -0.04931 -0.04931 0.04931 -0.05479 0.00000 -0.01644 $java Neural 400 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 -0.2 1.7 0 0.03910 -0.00782 0.06647 0.02606 -0.00521 0.04431 0.13032 0.28411 0.18636 $java Neural 400 4 3 2 1 0 -1 -2 -3 -4 -4 1 0 0.00000 0.00000 0.00000 -0.05006 0.20025 -0.05006 0.01252 0.00000 0.00000 5. Now we perform one step of stochastic gradient descent. With step size η, we update the weights: wi = wi − η ∂E ∂wi , i = 1 . . . 9. When FLAG=500, arg1 ... arg9 are the weights w1, . . . , w9, arg10=x1, arg11=x2, arg12=y, and arg13=η. Print four lines: (a) the old w1 . . . w9 (b) the error E under the old w (c) the updated w1 . . . w9 (d) the error E after the update For example, $java Neural 500 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 -1 1 0.1 0.10000 0.20000 0.30000 0.40000 0.50000 0.60000 0.70000 0.80000 0.90000 0.03778 0.10438 0.20438 0.29562 0.40493 0.50493 0.59507 0.70548 0.80000 0.90164 0.03617 $java Neural 500 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 -0.2 1.7 0 0.1 1.00000 0.90000 0.80000 0.70000 0.60000 0.50000 0.40000 0.30000 0.20000 0.31402 0.99609 0.90078 0.79335 0.69739 0.60052 0.49557 0.38697 0.27159 0.18136 0.29972 $java Neural 500 4 3 2 1 0 -1 -2 -3 -4 -4 1 0 0.1 4.00000 3.00000 2.00000 1.00000 0.00000 -1.00000 -2.00000 -3.00000 -4.00000 0.00710 4.00000 3.00000 2.00000 1.00501 -0.02002 -0.99499 -2.00125 -3.00000 -4.00000 0.00371 6. Long long ago, in a galaxy far far away, 117 students took an AI class. We have a data set where x1 is each student’s score on homework 2, x2 is their midterm score (both scaled to between 0 and 1), and the binary label y is whether they received an A at the end of semester (y = 1) or not (y = 0). We already randomly split the data set into a training set of n = 67 items, an evaluation set with 25 items, and a test set with 25 items. These files are denoted by the file names. We will train the neural network to predict if a student will receive an A at the end of semester. We first define one “epoch” as going over the training items once in the natural order: (a) FOR j = 1 TO n (b) wi = wi − η ∂Ej ∂wi , i = 1 . . . 9 (c) PRINT w’s evaluation set error (d) END CS 540 Fall 2017 where Ej is the error on the j-th training item. In other words, this is stochastic gradient descent (except that we go over training items in order rather than randomly, but we shall ignore the difference for now). Step (c) computes the evaluation set error as follows: X 25 k=1 1 2 (vC (xk, w) − yk) 2 where (xk, yk) is an evaluation set item, and we used the notation vC (xk, w) to mean the neural network output when the input is xk and the weights are w. When FLAG=600, arg1 ... arg9 are the initial weights w1, . . . , w9, arg10=η. Run one epoch (i.e. going over the 67 training items once). For each training item, print three lines: (a) x1, x2, y (b) the updated w1 . . . w9 (c) the evaluation set error under the updated w Therefore, you will print 67 ∗ 3 = 201 lines in total. For example, $java Neural 600 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 0.98000 0.87000 1.00000 0.10049 0.20048 0.30043 0.40055 0.50054 0.60048 0.70062 0.80034 0.90087 7.18915 0.92000 0.88000 0.00000 0.09486 0.19530 0.29548 0.39422 0.49471 0.59491 0.69358 0.79648 0.89110 7.12401 ... 0.81000 0.75000 0.00000 -0.13841 0.01138 0.16820 0.06903 0.24590 0.41273 0.22295 0.73651 0.52301 3.93300 0.94000 0.69000 0.00000 -0.13841 0.01138 0.16820 0.06135 0.23868 0.40744 0.20827 0.73651 0.51442 3.87761 7. We now run T epochs. For clarity, we will only print evaluation set error at the end of each epoch. When FLAG=700, arg1 ... arg9 are the initial weights w1, . . . , w9, arg10=η, arg11=T. At the end of each epoch, print w, then on a separate line the evaluation set error. $java Neural 700 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 3 -0.13841 0.01138 0.16820 0.06135 0.23868 0.40744 0.20827 0.73651 0.51442 3.87761 -0.15422 0.01050 0.18620 -0.10760 0.12065 0.33611 -0.14930 0.73903 0.41698 2.99174 -0.18275 0.01188 0.22380 -0.19706 0.06861 0.31792 -0.35328 0.74892 0.41936 2.71163 CS 540 Fall 2017 8. FLAG=800 has the same input as FLAG=700. But we will stop as soon as evaluation set error starts to increase after completing a whole epoch. If evaluation set error decreases or stays the same, we will step after T epochs. Print the number of epochs that actually happened, the final w, and the evaluation set error on different lines. Now you have a well-trained neural network. For a test item x, you can make a binary classification prediction by thresholding the network output at 1/2 (predict positive class if the output is exactly 1/2). Compute the test set classification accuracy (note: not the squared error) and print it, too. $java Neural 800 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 10000 788 -3.51459 1.20379 4.45346 -2.72310 0.93258 3.44859 -3.99435 2.11219 1.63381 0.87178 0.96000 $java Neural 800 -0.8 0.5 0 -1 0 -0.7 -0.7 0.9 0.1 0.1 10000 2 -0.80000 0.50000 0.00000 -1.00000 0.00000 -0.70000 -0.65318 0.90000 0.10000 2.56847 0.72000

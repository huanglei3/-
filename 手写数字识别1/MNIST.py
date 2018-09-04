import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

Input_Node = 784 #28*28
Output_Node = 10 #0-10

layer1_node = 500  #隐藏层节点数,只使用一层隐藏层
batch_size = 100   #一个训练批次的训练数据个数

learning_rate_base = 0.8 #初始学习率,后面会根据步长指数变化
learning_rate_decay = 0.99 #学习率的衰减率
Regularization_rate = 0.0001 #损失函数的系数lambda
Training_steps = 30000 #训练轮数
Moving_average_decay = 0.99 #滑动平均衰减率,为了增强数据的健壮性

"""
定义个函数,计算神经网络的前向传播结果
通过Relu激活函数的三层全连接神经网络,通过加入隐藏层实现多层网络结构
这样方便咋测试时使用滑动平均模型
"""
def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2):
    #当么有提供滑动平均类似,直接计算,不使用滑动平均模型
    if(avg_class==None):
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        #先使用ave_class.average计算出变量的滑动平均值,然后在计算前线传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        return tf.nn.relu(tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2))
#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32,shape=[None,Input_Node],name="x-input")#代表像素28*28
    y_ = tf.placeholder(tf.float32,shape=[None,Output_Node],name="y_input")#代表十个标签
    
    #生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal([Input_Node,layer1_node],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[layer1_node]))
    
    weight2 = tf.Variable(tf.truncated_normal([layer1_node,Output_Node],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[Output_Node]))
    
    #计算当前参数下的前向传播结果
    y = inference(x,None,weight1,biases1,weight2,biases2)
    global_step = tf.Variable(0,trainable=False)#不可训练变量
    
    #给定滑动平均衰减率和训练轮数的变量,初始化滑动平均类,给定训练轮数的变量可以加快早期变量的更新速度,即前期下降的快,后面慢慢收敛
    variable_averages = tf.train.ExponentialMovingAverage(Moving_average_decay,global_step)
    #decay*shadow_variable+(1-decay)*variable
    #每次的衰减率min(decay,(1+num_update)/(10+num_update))
    
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算使用了滑动平均之后的前向传播结果,
    #滑动平均不会改变变量本身的取值,只是用一个影子变量来记录滑动平均值,所以当需要使用这个滑动平均值的时候需明确调用average函数
    average_y = inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
    
    #计算交叉熵
#     cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+
#(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
                                
#     train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)#使用优化器进行优化,调最优参数
                                
    #在这里直接使用自带的函数spare_softmax_cross_entropy_with_logits来计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #argmax()参数第一个矩阵,axis axis=0按列比较每个向量里最大的数的index,axis=1返回按行比较的最大数的index
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #计算l2正则化的损失函数 j(theta)+lamdba*R(wi)  防止过拟合
    regularizer = tf.contrib.layers.l2_regularizer(Regularization_rate)
    #只计算权重,不用管偏执值
    regularization = regularizer(weight1)+ regularizer(weight2)
    
    loss = cross_entropy_mean + regularization
    
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,#初始的学习率
        global_step,#当前迭代的轮数
        mnist.train.num_examples / batch_size , #所有迭代次数
        learning_rate_decay #学习率衰减速度
    )
    #使用优化器优化损失函数(交叉熵,正则化)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step,variable_averages_op)#反向传播更新网络里的参数
    
    #averager_y ==> batch_size*10的矩阵 ,可以认为是真实值,,y_ 预测值,
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #判断两个张量的一维是否相等,相等为True,即预测的值
    
    #
    #将上述的布尔值转化为实数,然后计算平均值
    #即正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        validate_feed = {
            x:mnist.validation.images,
            y_:mnist.validation.labels
        }
        test_feed = {
            x:mnist.test.images,
            y_:mnist.test.labels
        }
        for i in range(Training_steps):
            #每一轮的训练数据
            xs,ys = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps ,validation accuracy is %g"%(i,validate_acc))
        #训练结束之后,在测试数据集上检测正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps ,validation accuracy is %g"%(Training_steps,test_acc))
        
def main(argv=None):
    mnist = input_data.read_data_sets("E:\MNIST_data",one_hot=True)
    train(mnist)
if __name__ =='__main__':
    tf.app.run()

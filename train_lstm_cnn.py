import tensorflow as tf
max_features=2000
embed_dim=128
hidden=128
iterations=200
import getdata
X_train, X_test, Y_train, Y_test=getdata.data_input()#获取数据

with tf.name_scope("input_x"):
    xs=tf.placeholder(tf.int32,[None,X_train.shape[1]])
with tf.name_scope("input_y"):
    ys=tf.placeholder(tf.float32,[None,2])
with tf.name_scope("Batch_size"):
    batch_size=tf.placeholder(tf.int32,[])
with tf.name_scope("keep_prob"):
    keep_prob=tf.placeholder(tf.float32,[])

with tf.device("/cpu:0"),tf.name_scope("embedding"):
    W=tf.Variable(tf.random_normal([max_features,embed_dim],-1.0,1.0),name="embedding_W")
    embed=tf.nn.embedding_lookup(W,xs)
    
    

lstm=tf.contrib.rnn.LSTMCell(hidden,state_is_tuple=True)
lstm_out,lstm_state=tf.nn.dynamic_rnn(lstm,embed,dtype=tf.float32)
#print("the lstm_out",lstm_out)
#print("the lstm_state",lstm_state)

#卷积需要3D
lstm_out_expanded=tf.expand_dims(lstm_out,-1)
#print(lstm_out_expanded)#Tensor("ExpandDims:0", shape=(?, 28, 128, 1), dtype=float32)

filters=[3,5,7]
pooled_outputs=[]
for i,filter_size in enumerate(filters):
    with tf.name_scope("conv-maxpool-%s"%filter_size):
        filter_shape=[filter_size,embed_dim,1,32]
        W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
        b=tf.Variable(tf.constant(0.1,shape=[32]),name="b")
        conv=tf.nn.conv2d(lstm_out_expanded,W,strides=[1,1,1,1],padding="VALID",name="conv")
        h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
        pooled=tf.nn.max_pool(h,ksize=[1,X_train.shape[1]-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
        pooled_outputs.append(pooled)

        
num_filters_total=32*len(filters)
#print("the num_filters_total is %d"%num_filters_total)
h_pool=tf.concat(pooled_outputs,3)
#print("the h_pool",h_pool)
h_pool_flat=tf.reshape(h_pool,[-1,num_filters_total])
#print("the h_pool_flat",h_pool_flat)
with tf.name_scope("dropout"):
    h_drop=tf.nn.dropout(h_pool_flat,keep_prob)
with tf.name_scope("output"):
    W=tf.get_variable("W",
                     shape=[num_filters_total,2],
                     initializer=tf.contrib.layers.xavier_initializer())
    b=tf.Variable(tf.constant(0.1,shape=[2]),name="b")
    pred=tf.nn.xw_plus_b(h_drop,W,b,name="out")
            
        
with tf.name_scope("cross_entropy"):
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=ys))
tf.summary.scalar("cross_entropy",cross_entropy)
train=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(ys,1))
with tf.name_scope("accuracy"):
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
tf.summary.scalar("accuracy",accuracy)

batch=32
num_batch=X_train.shape[0]//batch
count=1
merged=tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer=tf.summary.FileWriter("train",sess.graph)
    for step in range(10000):
        for j in range(num_batch):
            batchx=X_train[j*batch:(j+1)*batch]
            batchy=Y_train[j*batch:(j+1)*batch]
            summary,_=sess.run([merged,train],feed_dict={xs:batchx,ys:batchy,keep_prob:0.9})
            train_writer.add_summary(summary,count)
            count+=1
            #sess.run(train,feed_dict={xs:batchx,ys:batchy,keep_prob:0.9})
        if step %100==0:
            _,loss,acc=sess.run([train,cross_entropy,accuracy],feed_dict={xs:X_train,ys:Y_train,keep_prob:1.0})
            print("%dth the loss is %f and the accuracy is %f in the train data"%(step,loss,acc))
    _,loss,acc=sess.run([train,cross_entropy,accuracy],feed_dict={xs:X_test,ys:Y_test,keep_prob:1.0})
    print("%dth the loss is %f and the accuracy is %f in the test data"%(step,loss,acc))
            
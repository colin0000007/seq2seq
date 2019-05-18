#encoding=UTF-8
import tensorflow as tf
#tensorflow.python.ops.rnn_cell_impl.LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import os
import time
'''
@author: outsider
@note: attention的seq2seq模型，序列到序列，使用encoder和decoder均使用rnn
可选的bidirectional rnn
可选的cell类型 LSTM BasicLSTM
可选的解码方法:greedy或者beam_search
'''
class AttentionSeq2SeqModel():
    #model_path指的是已经训练好的模型的地址，也就是说这时候是使用模型。而不是训练模型,如果训练模型暂时不需要指定model_path
    #除了model_path之外的其他参数是训练用到的参数，如果要训练必须传入
    #decode_method:解码方法，默认贪婪法:greedy，可选的还有beamsearch
    #如果使用beam search需要指定beam width 默认为10
    
    def __init__(self,model_path=None,*,src_vocab_size=None,tgt_time_step=None,tgt_vocab_size=None,start_token_id=None,end_toekn_id=None,attention_mechanism="normed_bahdanau",batch_size = None):
        #1.如果传入模型路径，尝试加载模型
        if model_path != None:
            self.model_path = model_path
            self.load_model()
            return
        #2检测训练参数
        if src_vocab_size == None or tgt_time_step == None or tgt_vocab_size == None or start_token_id == None or end_toekn_id == None:
            print("some of given training parameters are None!")
            exit(0)
        #所给参数正确
        self.src_vocab_size = src_vocab_size
        self.tgt_time_step = tgt_time_step
        self.tgt_vocab_size = tgt_vocab_size
        self.start_token_id = start_token_id
        self.end_token_id = end_toekn_id
        self.batch_size = batch_size
        attention_mechanism = attention_mechanism.lower()
        self.attention_mechanism = attention_mechanism
        if attention_mechanism != 'scaled_luong' and attention_mechanism != "normed_bahdanau":
            print("no such attention mechanism!")
            print("attention mechanism must be 'scaled_luong' or 'normed_bahdanau'!")
            exit(0)
        print("BasicSeq2SeqModel:")
        print("tgt_time_step:",self.tgt_time_step)
        print("src_vocab_size:",self.src_vocab_size)
        print("tgt_vocab_size:",self.tgt_vocab_size)
    #加载模型
    def load_model(self):
        self.inference_graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.inference_graph)
        dir,file = os.path.split(self.model_path)
        # 导入的计算图是inference的
        loader = tf.train.import_meta_graph(dir+'/inference/'+file+".inference"+ '.meta')
        #导入的数据是训练图的
        loader.restore(self.sess, self.model_path)
        
    def build_input_tensor(self,src_embedding_size,tgt_embedding_size):
        with tf.name_scope("input_tensor"):
            #这里设置位None，意思是不指定batch_size用到多少就是多少
            self.src_batch = tf.placeholder(tf.int32,[None,None],name="source_batch")
            #对source做词嵌入，词向量矩阵的shape为[source的词库大小,嵌入维度]
            #嵌入矩阵中每一行就是一个词向量
            src_embedding = tf.get_variable(shape=[self.src_vocab_size,src_embedding_size],name='source_embedding')
            #使用embedding_lookup从embedding矩阵中查询词向量从而将X的每一个单词的index转换一个词向量
            self.embedded_src_batch = tf.nn.embedding_lookup(src_embedding,self.src_batch)
            #最后返回的embedded_X.shape = [time_step,batch_size,embedding_size]
            #target类似
            self.tgt_batch_x = tf.placeholder(tf.int32,[None,None],name="target_batch_x")
            self.tgt_batch_y = tf.placeholder(tf.int32,[None,None],name="target_batch_y")
            self.tgt_embedding = tf.get_variable(shape=[self.tgt_vocab_size,tgt_embedding_size],name='target_embedding')
            self.embedded_tgt_batch_x = tf.nn.embedding_lookup(self.tgt_embedding,self.tgt_batch_x)
            self.src_batch_seq_len = tf.placeholder(tf.int32,[None],name="source_batch_seq_len")
            self.tgt_batch_seq_len = tf.placeholder(tf.int32,[None],name="target_batch_seq_len")
            #保存当前batch的最长序列值，mask的时候需要用到
            self.tgt_batch_max_len = tf.placeholder(tf.int32,[],name="target_batch_max_len")
    
    #只用于inference阶段的tensor
    def build_input_tensor_inference(self):
        #tf.name_scope对于get_variable并不会加上前缀
        with tf.variable_scope("input_tensor_inference"):
            #测试时输入数据的batch是多少，动态的传入，避免预测时必须固定batch
            self.inference_batch_size = tf.placeholder(dtype=tf.int32, shape=[1],name="inference_batch_size")
            #tf.tile将常量重复shape次数连接在一起
            self.start_tokens = tf.tile(tf.constant(value=self.start_token_id, dtype=tf.int32,shape=[1]), multiples = self.inference_batch_size, name="start_tokens")
    
    
    #获取rnn layer，指定layer层数和num_units
    def get_rnn_layer(self,layer_size,num_units,cell_type="LSTM"):
        def get_rnn_cell():
            if cell_type.lower() == "lstm":
                return tf.nn.rnn_cell.LSTMCell(num_units=num_units)
            elif cell_type.lower() == "basiclstm":
                return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
            elif cell_type.lower() == "gru":
                return tf.nn.rnn_cell.GRUCell(num_units=num_units)
            else:
                print("there is no this kind of cell type!")
                exit(0)
        return tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(layer_size)])
    def build_encoder(self,encoder_rnn_layer_size,encoder_num_units,encoder_cell_type="LSTM"):
        #定义encoder的rnn_layer
        with tf.name_scope("encoder_rnn"):
            rnn_layer = self.get_rnn_layer(encoder_rnn_layer_size, encoder_num_units, encoder_cell_type)
            #将rnn沿时间序列展开
            #   encoder_outputs: [batch_size,time_step, num_units]
            #   encoder_state: [batch_size, num_units]
            #   sequence_length:传入一个list保存每个样本的序列的真实长度，教程中说这样做有助于提高效率
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
            rnn_layer, self.embedded_src_batch,
            sequence_length=self.src_batch_seq_len, time_major=False,dtype=tf.float32)
            #注意返回的encoder_outputs返回了每一个序列节点的output shape为[max_time, batch_size, num_units]
            print("encoder_states:",encoder_states)
            print("encoder_outputs:",encoder_outputs)
            print("len(encoder_states):",len(encoder_states))
            #关于encoder_states：一个tuple，有多少个cell，元组的size就是多少，保存了每一个cell运行后的c和h值
        return encoder_outputs, encoder_states
    
    '''
            双向rnn的encoder
            参照Google官方的nmt教程
    '''
    def build_encoder_bi(self,encoder_rnn_layer_size,encoder_num_units,encoder_cell_type="LSTM"):
        encoder_cell_type = encoder_cell_type.lower()
        #定义encoder的rnn_layer
        with tf.name_scope("encoder"):
            fw_rnn_layer = self.get_rnn_layer(encoder_rnn_layer_size, encoder_num_units, encoder_cell_type)
            bw_rnn_layer = self.get_rnn_layer(encoder_rnn_layer_size, encoder_num_units, encoder_cell_type)
            #双向rnn展开
            '''
                bi_state的结构:(fw_state,bw_state)
                                fw_state=((c,h),(c,h)...)
                                bw_state = ((c,h),(c,h)...)
            '''
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_rnn_layer, bw_rnn_layer, self.embedded_src_batch,
            sequence_length=self.src_batch_seq_len, time_major=False,dtype=tf.float32)
            '''                    
                                将前向rnn和后向rnn的output的最后一个维度连接起来,比如 fw:[128,10,100],bw:[128,10,100],那么连接后为[128,10,200]
                                这样导致一个问题就是encoder rnn的输出和decoder rnn的输入对应不上了，encoder因为拼接了fw和bw变成了200，
                                有  2个解决办法，将encoder rnn的num_units变成decoder的一半或者反过来将decoder rnn的num_units增大一倍
            '''
            encoder_outpus = tf.concat(bi_outputs, -1)
            '''
                bi_state同样有fw和bw，怎样拼接在一起呢？
                    (1)参照output的拼接直接，拼接c和h的最后一个维度：这种方法问题在于tuple必须是特殊类型的tuple,比如LSTMStateTuple
                    (2)直接将fw和bw的结果堆叠在一起这样cell的个数相当于翻了一倍，需要调整encoder或者decoder的cell个数
                                        另外使用不同的cell也有区别的，LSTM有c和h，而GRU只有一个值。
            '''
            fw_encoder_state = bi_state[0]
            bw_encoder_state = bi_state[1]
            encoder_states = []
            if encoder_cell_type == "lstm" or encoder_cell_type == "basiclstm":
                #i循环cell的个数
                for i in range(encoder_rnn_layer_size):
                    #连接当前cell fw和bw的c，h
                    c = tf.concat([fw_encoder_state[i][0],bw_encoder_state[i][0]],-1)
                    h = tf.concat([fw_encoder_state[i][1],bw_encoder_state[i][1]],-1)
                    encoder_states.append(LSTMStateTuple(c,h))
            else:#GRU
                #state中每个cell只有一个值
                for i in range(encoder_rnn_layer_size):
                    state = tf.concat([fw_encoder_state[i],bw_encoder_state[i]],-1)
                    encoder_states.append(state)
            encoder_states = tuple(encoder_states)
        print("bidirectional encoder-encoder_outputs:",encoder_outpus)
        print("bidirectional encoder-encoder_states:",encoder_states)
        return encoder_outpus,encoder_states
    #构造训练时的decoder
    def build_decoder(self,encoder_states,encoder_outputs,decoder_rnn_layer_size,decoder_num_units,decoder_cell_type="LSTM"):
        #decoder的rnn layer
        with tf.variable_scope("decoder"):
            self.decoder_rnn_layer = self.get_rnn_layer(decoder_rnn_layer_size, decoder_num_units, decoder_cell_type)
            #加入attention
            self.decoder_rnn_layer_attention = self.build_rnn_layer_attention(encoder_outputs, self.decoder_rnn_layer, decoder_num_units,self.src_batch_seq_len)
            #decoder：需要helper，rnn cell，helper被分离开可以使用不同的解码策略，比如预测时beam search，贪婪算法
            #这里projection_layer就是一个全连接层，encoder_output的维度不能和target词汇数量一致所以需要映射层
            #为什么这里projection_layer不指定激活函数为softmax，最后构建loss的传入的是logits，我的理解logits是没有经过激活函数的
            #的值，logits = W*X+b
            self.projection_layer = tf.layers.Dense(units=self.tgt_vocab_size,kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
            #训练使用的training_helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                self.embedded_tgt_batch_x, self.tgt_batch_seq_len, time_major=False)
            #加入attention后需要加入init_state取代原来的encoder_states
            init_state = self.decoder_rnn_layer_attention.zero_state(self.batch_size, tf.float32).clone(
                cell_state=encoder_states)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_rnn_layer_attention, training_helper, init_state,
                output_layer=self.projection_layer)
        #将序列展开
        '''
         impute_finished=False,
        maximum_iterations=None,
        swap_memory=False , Whether GPU-CPU memory swap is enabled for this loop.
        '''
        decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,swap_memory = True)
        #decoder_output的shape:[batch_size,序列长度,tgt_vocab_size]
        print("decoder_output:",decoder_output)
        return decoder_output
    
    def build_rnn_layer_attention(self,encoder_output,decoder_layer,decoder_num_units,src_seq_len):
        #scaled_luong  normed bahdanau
        attention = None
        if self.attention_mechanism == 'scaled_luong':
            attention = tf.contrib.seq2seq.LuongAttention(
                decoder_num_units, encoder_output,
                memory_sequence_length=src_seq_len,scale=True)
        else:
            attention = tf.contrib.seq2seq.BahdanauAttention(decoder_num_units,
                            encoder_output, src_seq_len, normalize = True)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_layer, attention,
            attention_layer_size=decoder_num_units,name='attention')
        return decoder_cell
   
    #构造推理时的decoder
    def build_decoder_inference(self,encoder_outputs,encoder_states,beam_width,decoder_num_units,decoder_rnn_layer_size,decoder_cell_type,decoding_method):
        decoding_method = decoding_method.lower()
        decoder = None
        with tf.variable_scope("decoder"):
            rnn_layer = self.get_rnn_layer(decoder_rnn_layer_size, decoder_num_units, decoder_cell_type)
            projection_layer = tf.layers.Dense(units=self.tgt_vocab_size,kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
            #beam search解码
            if decoding_method == 'beamsearch':
                if beam_width <= 1:
                    raise Exception('the beam width must greater than 1')
                memory = tf.contrib.seq2seq.tile_batch(
                  encoder_outputs, multiplier=beam_width)
                src_seq_len = tf.contrib.seq2seq.tile_batch(
                      self.src_batch_seq_len, multiplier=beam_width)
                encoder_state = tf.contrib.seq2seq.tile_batch(
                      encoder_states, multiplier=beam_width)
                batch_size = self.inference_batch_size * beam_width
                inference_rnn_layer_attention = self.build_rnn_layer_attention(memory, rnn_layer, decoder_num_units,src_seq_len)
                decoder_initial_state = inference_rnn_layer_attention.zero_state(batch_size, tf.float32).clone(
                    cell_state=encoder_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=inference_rnn_layer_attention,
                        embedding=self.tgt_embedding,
                        start_tokens=self.start_tokens,
                        end_token=self.end_token_id,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=projection_layer,
                        length_penalty_weight=1.0,
                        coverage_penalty_weight=0.0)
            #greedy解码
            elif decoding_method == 'greedy':
                rnn_layer_attention = self.build_rnn_layer_attention(encoder_outputs, rnn_layer, decoder_num_units, self.src_batch_seq_len)
                predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.tgt_embedding,self.start_tokens,self.end_token_id)
                #encoder_states替换为init_state
                init_state = rnn_layer_attention.zero_state(self.inference_batch_size, tf.float32).clone(
                    cell_state=encoder_states)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    rnn_layer_attention, predicting_helper, init_state,
                    output_layer=projection_layer)
            else:
                raise Exception('不支持的解码方法，只可选 greedy 或者 beam search/Unsupported decoding method, only greedy or beamsearch are allowed!')
        # Dynamic decoding
        predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,maximum_iterations=self.tgt_time_step,swap_memory = True)
        #predicted_ids：返回shape:[batchsize,max_len,beam_width]
        if decoding_method == 'beamsearch':
             #predicted_ids保存了所有结果，从好到坏排名，这里直接取最好的
            tf.identity(predicting_decoder_output.predicted_ids[:,:,0], name='predictions')
        else:
            tf.identity(predicting_decoder_output.sample_id, name='predictions')
    def seq_loss(self,rnn_output):
        batch_size = 128
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_output,
                labels=self.tgt_batch_y)
        loss_mask = tf.sequence_mask(self.tgt_batch_seq_len,self.tgt_time_step,dtype=tf.float32,name="masks")
        cost = cost * tf.to_float(loss_mask)
        return tf.reduce_sum(cost) / tf.to_float(batch_size)
        
    #构建训练计算图和推理计算图
    #解决attention参数共享问题
    def build_graph(self,src_embedding_size,tgt_embedding_size,is_encoder_bidirectional=False,rnn_layer_size=2, rnn_num_units=128,cell_type="LSTM",lr=0.001,decoding_method="greedy",beam_width=10):
        #构造计算图
        train_graph = tf.Graph()
        inference_graph = tf.Graph()
        #必须先构建推理图，后面创建的input tensor覆盖inference的，因为训练中需要用到train graph中的图
        with inference_graph.as_default():
            #1.tensor声明
            self.build_input_tensor(src_embedding_size, tgt_embedding_size)
            self.build_input_tensor_inference()
            #2.encoder decoder inference
            if is_encoder_bidirectional:
                #需要注意使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
                encoder_outputs, encoder_states = self.build_encoder_bi(rnn_layer_size, rnn_num_units/2, cell_type)
            else:
                encoder_outputs, encoder_states = self.build_encoder(rnn_layer_size, rnn_num_units, cell_type)
            #选择合适的解码方法
            self.build_decoder_inference(encoder_outputs, encoder_states, beam_width, rnn_num_units, rnn_layer_size, cell_type, decoding_method)
            #inference部分不需要loss，optimize的op
            #构建训练图
            with train_graph.as_default():
                #1.tensor声明
                #train阶段的tensor作为了class的属性
                self.build_input_tensor(src_embedding_size, tgt_embedding_size)
                #2.encoder decoder inference
                if is_encoder_bidirectional:
                    #需要注意使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
                    encoder_outputs, encoder_states = self.build_encoder_bi(rnn_layer_size, rnn_num_units/2, cell_type)
                else:
                    encoder_outputs, encoder_states = self.build_encoder(rnn_layer_size, rnn_num_units, cell_type)
                decoder_outputs = self.build_decoder(encoder_states,encoder_outputs,rnn_layer_size, rnn_num_units, cell_type)
                #train graph不需要inference阶段的解码过程
                #对训练输出取名字
                training_logits = tf.identity(decoder_outputs.rnn_output, 'logits')
                print("training_logits.shape:",training_logits.shape)
                print("tgt_batch_y.shape:",self.tgt_batch_y.shape)
                #mask的作用是：计算loss时忽略pad的部分，这部分的loss不需要算，
                masks = tf.sequence_mask(self.tgt_batch_seq_len, self.tgt_batch_max_len, dtype=tf.float32, name='masks')
                with tf.name_scope("optimization"):
                    #loss
                    self.loss = tf.contrib.seq2seq.sequence_loss(
                        training_logits,
                        self.tgt_batch_y,
                        masks)
                    #self.loss = self.seq_loss(training_logits)
                    optimizer = tf.train.AdamOptimizer(lr)
                    # Gradient Clipping
                    gradients = optimizer.compute_gradients(self.loss)
                    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                    self.train_op = optimizer.apply_gradients(capped_gradients)
        return train_graph,inference_graph
    
    #训练
    def train(self,model_path,generator,src_embedding_size,tgt_embedding_size,is_encoder_bidirectional=False,rnn_layer_size=2, rnn_num_units=128, cell_type="LSTM",lr=0.001,decoding_method="greedy",beam_width=10):
        train_graph,inference_graph = self.build_graph(src_embedding_size, tgt_embedding_size,is_encoder_bidirectional,rnn_layer_size, rnn_num_units, cell_type, lr,decoding_method,beam_width)
        #先保存空的inference graph
        with tf.Session(graph=inference_graph) as sess:
            #这里需要初始化tensor吗，问题是tf.initglobal会初始化所有计算图中的tensor吗
            #未知，暂时写
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            dir,file = os.path.split(model_path)
            #创建inference文件夹保存inference计算图
            if not os.path.exists(dir+"/inference"):
                os.mkdir(dir+"/inference")
            saver.save(sess,dir+"/inference/"+file+".inference")
            #writer = tf.summary.FileWriter("./", sess.graph)
            #writer.close()
        with tf.Session(graph=train_graph) as sess:
            #tensor初始化
            sess.run(tf.global_variables_initializer())
            #writer = tf.summary.FileWriter("./", sess.graph)
            #writer.close()
            #获取数据生成器
            for src_batch,tgt_batch_x,tgt_batch_y,tgt_batch_max_len,src_batch_seq_len,tgt_batch_seq_len,batch_num,epoch_num in generator:
                b_loss,_ = sess.run([self.loss,self.train_op],feed_dict={
                    self.src_batch:src_batch
                    ,self.tgt_batch_x:tgt_batch_x
                    ,self.tgt_batch_y:tgt_batch_y#np.array(tgt_batch_y).reshape(target_max_len,batch_size)
                    ,self.src_batch_seq_len:src_batch_seq_len
                    ,self.tgt_batch_seq_len:tgt_batch_seq_len
                    ,self.tgt_batch_max_len:tgt_batch_max_len
                    })
                print("epoch:",epoch_num," batch:",batch_num,"loss:",b_loss)
            
            # 保存模型
            saver = tf.train.Saver()
            saver.save(sess, model_path)
        #加载模型
        #self.model_path = model_path
        #self.load_model()
    #预测
    #input: [batch_size,time_step]
    #input_seq_len:[batch_size]
    def predict(self,input,input_seq_len):
        source_batch_input = self.inference_graph.get_tensor_by_name('input_tensor/source_batch:0')
        logits = self.inference_graph.get_tensor_by_name('predictions:0')
        input_batch_size = self.inference_graph.get_tensor_by_name("input_tensor_inference/inference_batch_size:0")
        src_seq_len = self.inference_graph.get_tensor_by_name('input_tensor/source_batch_seq_len:0')
        answer_logits = self.sess.run(logits,{source_batch_input: input, 
                                          src_seq_len: input_seq_len,
                                          input_batch_size:[len(input)]
                                          })
        #不需要关闭，
        #self.sess.close()
        return answer_logits

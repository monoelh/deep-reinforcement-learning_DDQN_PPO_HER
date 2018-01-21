#######################################################################################
# Neural network framework to play around with
# Author: Manuel Hass
# 2017
# 
# *examples in the end
#######################################################################################
try:
    import numpy as np
    numpy = np
except ImportError:
    print ('ERROR -> MODULE MISSING: numpy ')


###################### loss ####################################################
def bce(ya,yta,dev=False):  ############ not robust !!
    '''
        binary cross entropy
    '''
    if (dev==True):
        return (yta-ya)/((1-yta)*yta)
    return -(np.sum(ya*np.log(yta)+(1.-yta)*np.log(1.-yta))/(yta.shape[0]*2.0))

def qef(ya,yta,dev=False):
    '''
        quadratic error function ||prediction-target||²
    '''
    if (dev==True):
        return (yta-ya)
    return np.sum((yta-ya)**2)/(yta.shape[0]*2.0)

def phl(y,yt,dev=False,delta=1.):
    '''
        subquadratic error function (pseudo huber loss)
    '''
    a = (yt-y)
    if (dev==True):
        return  a/( np.sqrt(a**2/delta**2 +1) ) 
    return np.sum((delta**2)*(np.sqrt(1+(a/delta)**2)-1)/(yt.shape[0]*2.0))


###################### regularization ####################################################
def L2_norm(lam,a):  
    '''
        2-Norm regularizer
    '''
    return lam*a

def L1_norm(lam,a):
    '''
        1-Norm regularizer
    '''
    return lam*np.sign(a)


###################### activation  ####################################################
def f_lgtr(a,dev=False):
    '''
        (robust) logistic transfer function 
            sigmoidal [0,1]
    '''
    if (dev==True):
        return (1-np.tanh(a/2.)**2)/2.
    return  (np.tanh(a/2.)+1)/2.

 
def f_stoch(a,dev=False):
    '''
        stochastic transfer function 
            activates if activated input > ~Uniform
            binary [0,1]
    '''
    if (dev==True):
        return np.zeros(a.shape)  
    x = f_lgtr(a,dev=False)
    rand = np.random.random(x.shape)
    return  np.where(rand < x,1,0)

def f_tanh(a,dev=False):
    '''
        hyperbolic tangent transfer function 
            sigmoidal [-1,1]
    '''
    if (dev==True):
        return (1-np.tanh(a)**2)
    return  np.tanh(a)

def f_atan(a,dev=False):
    '''
        arcus tangent transfer function 
            sigmoidal [-pi/2, pi/2]
    '''
    if (dev==True):
        return (1/(a**2+1))
    return  np.arctan(a)

def f_sp(a,dev=False):
    '''
        softplus transfer function 
            [0,a]

            ### kinda clip it...to make more robust
    '''
    if (dev==True):
        return np.exp(a)/(np.exp(a)+1.)
    return  np.log(np.exp(a)+1.)
    
def f_relu(a,dev=False):
    '''
        rectified linear transfer function 
            [0,a]
    '''
    if (dev==True):
        return np.maximum(0,np.sign(a)) 
    return  np.maximum(0.0,a)
 
def f_bi(a,dev=False):
    '''
        bent identity transfer function
    '''
    if (dev==True):
         return a / ( 2.0*np.sqrt(a**2+1) ) + 1
    return  (np.sqrt(a**2+1)-1)/2.0 + a

def f_iden(a,dev=False):
    '''
        identity transfer function 
    '''
    if (dev==True):
         return np.ones(a.shape)
    return  a

def f_bin(a,dev=False):
    '''
        binary step transfer function 
    '''
    if (dev==True):
         return np.zeros(a.shape) 
    return  np.sign(f_relu(a))


############################# utils ######################################
### input / output processing
def one_hot(targets,smooth=False):
    '''
        input: discrete labels (number, string, etc.)
        output: binary numpy array (size = #unique classes)
    '''
    classes =  np.unique(targets.T)
    binarycoded = []
    for i in classes:
        binarycoded +=  [np.where(targets==i,1,0)[0]]
    out = np.array(binarycoded).T
    if smooth:
        # one side label smoothind
        out = out+.8 +.1
    else:
        return out

def hot_one(targets):
    '''
        input: binary array
        output: discrete labels (numbers)
    '''
    return np.argmax(np.array(targets).T,axis=0).reshape(-1,1)


############################ MLP ########################################### 
### network layer
class layer:
    '''
    actiavtion layer for model building:
        layer(input_dimension,number_of_nodes)

    parameters:
        f   : activation function
        w   : weights

        reg : regularizer function
        lam : regularizer lambda
        eta : learning rate
        
        opt : optimizer ('Adam','RMSprop','normal')
        eps : "don't devide by zero!!"
        b1  : momentumparameter for 'Adam' optimizer
        b2  : momentumparameter for 'RMSprop' and 'Adam' optimizer
        m1  : momentum for 'Adam' optimizer
        m2  : momentum for 'RMSprop' and 'Adam' optimizer

        count: number of updates 

    '''
    def __init__(self,in_dim,nodes=32,no_bias=False): 

        #activation and weights
        self.no_bias = no_bias
        self.f = f_relu
        if self.no_bias: self.w = np.random.uniform(-.1,.1,(nodes,in_dim))
        else: self.w = np.random.uniform(-.1,.1,(nodes,in_dim+1))

        #momentum 
        self.m1 = np.random.uniform(0.1,1,self.w.shape)
        self.m2 = np.random.uniform(0.1,1,self.w.shape)
        self.b1 = 0.9   # Adam, if b1 = 0. -> Adam = RMSprop
        self.b2 = 0.999
        self.opt = 'Adam'
        self.eps = 1e-8

        #regularizer
        self.reg = L2_norm
        self.lam = 0#1e-5

        #learning
        self.count = 0
        self.eta = 1e-4

    def forward(self,input_):
        '''
            forward pass (computes activation)
                return: activation(input * weights[+ bias])
        '''
        ##### IF no_bias != True  :
        if self.no_bias: self.x1 = input_
        else: self.x1 = np.vstack((input_.T,np.ones(input_.shape[0]))).T
        #print('fw x1: ',self.x1.shape)
        self.h1 = np.dot(self.x1,self.w.T).T
        #print('fw h1: ',self.h1.shape)
        self.s = self.f(self.h1)
        #print('fw s: ',self.s.shape)
        return self.s

    def backward(self,L_error):
        '''
            backward pass (computes gradient)
                return: layer delta
        '''
        #print('L_error :  ',L_error.shape)
        self.L_grad = L_error* self.f(self.h1,True).T
        #print('L_grad :  ',self.L_grad.shape)
        self.delta_W = -1./(self.x1).shape[0] * np.dot(self.L_grad.T,self.x1) - self.reg(self.lam,self.w)
        return np.dot(self.w.T[1:],self.L_grad.T).T

    def update(self):   
        '''
            update step (updates weights & momentum)
        '''   
        self.m1 = self.b1*self.m1 + (1-self.b1)*self.delta_W
        self.m2 = self.b2*self.m2 + (1-self.b2)*self.delta_W**2
        if(self.opt=='RMSprop'):
            self.w += self.eta* self.delta_W / (np.sqrt(self.m2) +self.eps)
        if (self.opt=='Adam'):
            self.w += self.eta* self.m1 / (np.sqrt(self.m2) +self.eps)
        if(self.opt=='normal'):
            self.w += self.eta* self.delta_W
        self.count += 1

    def reset(self):
        '''
            weights & momentum reset
        '''
        self.w = np.random.uniform(-.7,.7,(nodes,in_dim+1))
        self.m1 = np.random.uniform(0.,1,self.w.shape)
        self.m2 = np.random.uniform(0.,1,self.w.shape)


### (dropout layer) ## need to specify if not training... w*(1-droptrate)
class dropout:
    '''
        masks activations
            dropout(input,drop)

        parameter:
            drop    : chance for dropping unit

    '''
    def __init__(self,in_dim,drop =.5,training=True): 
        self.training = training
        # dropout mask
        self.drop = drop
        self.mask = np.random.choice([0, 1], size=(in_dim), p=[self.drop, 1-self.drop])


    def forward(self,input_):
        '''
            masks input
        ''' 
        if not self.training: return (1.-self.drop)*input_.T
        return (self.mask*input_).T

    def backward(self,L_error):
        '''
            masks backward pass 
        '''
        return self.mask * L_error

    def update(self):   
        '''
            updates mask
        '''   
        self.mask = np.random.choice([0, 1], size=(self.mask.shape[0]), p=[self.drop, 1-self.drop])

    def reset(self):
        '''
            also updates mask
        '''   
        self.update()


### models
class mlp:
    '''
    multi layer perceptron model:
        mlp(List_with_layers)

    parameters:
        Layerlist   : list of layers
        erf         : errorfunction
        loss        : last training loss
        
    '''
    def __init__(self,Layerlist):
        self.Layerlist = Layerlist
        self.erf = qef

    def infer(self, input_):
        '''
            compute full forward pass
        '''            
        out = input_
        for L in self.Layerlist:
            out = L.forward(out).T
        return out

    def train(self,input_,target_):
        '''
            training step
        '''
        self.loss = self.erf(target_,self.infer(input_))       
        grad = self.erf(target_,self.infer(input_),True)
        for L in self.Layerlist[::-1]:
            #print(grad.shape, 'before grad')
            grad = L.backward(grad)
            #print(grad.shape, 'after grad')

            L.update()

class dense_mlp:
    '''
    dense multi layer perceptron model:
        dense_mlp(List_with_layers)

    parameters:
        Layerlist   : list of layers
        erf         : errorfunction
        loss        : last training loss
        
    '''
    def __init__(self,Layerlist):
        self.Layerlist = Layerlist
        self.erf = qef

    def infer(self, input_):
        '''
            compute full forward pass
            ---with dense looped connections !!!!!!!!!!!!!!!!
        '''            
        out = input_
        out_list = [out] #list holding all layers activation
        L = self.Layerlist
        for i in range (len (L)):
            #print('before shape ',out.shape)
            #print(out_list[0].shape)
            out = np.hstack((out_list[j] for j in range(i+1)))
            
            #print('after shape ',out.shape)
            out = L[i].forward(out).T
            #print('out ',out.shape)
            out_list += [out]
            #print(out_list[0].shape,out_list[1].shape)
    
        return out

    def train(self,input_,target_):
        '''
            training step
            ---nested backprop 
        '''
        self.loss = self.erf(target_,self.infer(input_))       
        grad = self.erf(target_,self.infer(input_),True)
        L = self.Layerlist
        #print('round ...........')
        for i in range(len(L)):
            grad = L[-(i+1)].backward(grad)
            L[-(i+1)].update()
            grad_list = grad[:,:-L[-(i+1)].w.shape[0]]
            if len(L) > (i+2):
                for j in (np.arange(len(L)-(i+2))):
                        grad_ = grad_list[:,:L[j].w.shape[0]]
                        grad_list = grad_list[:,L[j].w.shape[0]:]
                        _ = L[j].backward(grad_)
                        L[j].update()

            grad = grad[:,-L[max(-(i+2),-len(L))].w.shape[0]:]  

class dueling_mlp:
    ''' !!!!!!!!!!!!!!!! NOT WORKING YET !!!!!!!!!!!!!!!!
    - not converging so far - think it's about target_updates:
      double Q update :  Q_target = reward + gamma * tagetNet(state+1)[0,argmax(onlineNet(state+1))]
      dueling Q update:  Q_target = reward + gamma * 1/number_of_actions * mean of maybe all Q action values?!
    dueling mlp for Q learning:
        dueling_mlp(LL0,LLA,LLB,model=mlp)
        IN -> LL0 -> [LLV & LLA] => (LLV + (LLA-mean(LLA)))
    parameters:
        LL0         : list of layers for core model
        LLV         : list of layers for value model (shape 1)
        LLA         : list of layers for advantage model (shape actionspace)

    '''
    def __init__(self,LL0,LLV,LLA):
        self.LL0 = LL0
        self.LLV = LLV
        self.LLA = LLA
        self.erf = qef
        

    def infer(self, input_):
        '''
            compute full forward pass over both networks
        '''      
        out0 = input_
        for L in self.LL0:
            out0 = L.forward(out0).T
        outV = out0
        outA = out0
        for L in self.LLA:
            outA = L.forward(outA).T
        for L in self.LLV:
            outV = L.forward(outV).T
       
        outA_ = outA-outA.mean(0)
        outQ = outA_ + outV

        return outQ

    def train(self,input_,target_):
        '''
            training step
            think about the aggregation layer in the end ---> LLout maybe
        '''
        # calculating forward pass
        out0 = input_
        for L in self.LL0:
            out0 = L.forward(out0).T
        outV = out0
        outA = out0
        for L in self.LLA:
            outA = L.forward(outA).T
        for L in self.LLV:
            outV = L.forward(outV).T
        outA_ = outA-outA.mean(0)
        outQ = outA_ + outV
       

        self.TD_loss = np.power((target_-outQ),2) #this is TD error. use it for prioritized experience replay sampling      
        #print(self.TD_loss.shape,'TD loss shape')
        self.loss = self.erf(target_,outQ) # or use this as single value
        #print(self.loss.shape,'TD loss shape')
        
        ###################################################
        gradA = self.erf(target_-target_.mean(1)[:,True],outA,True)
        gradV = self.erf(target_.mean(1)[:,True],outV,True)
        # this might be the reason why it's not working. calculate a correct loss please...
        ###################################################
        #gradA = self.TD_loss 
        #gradV = self.TD_loss.mean(1)[:,True]
        #print(gradV.shape, 'grad V')
        for L in self.LLA[::-1]:
            #print(grad.shape, 'before grad')
            gradA = L.backward(gradA)
            #print(grad.shape, 'after grad')
            L.update()
        for L in self.LLV[::-1]:
            #print(grad.shape, 'before grad')
            gradV = L.backward(gradV)
            #print(grad.shape, 'after grad')
            L.update()
        # 2 gradients arriving in feature extractor LL0
        # update LL0 with the mean of both
        gradA = .5 * gradA
        gradV = .5 * gradV
        for L in self.LL0[::-1]:
            #print(grad.shape, 'before grad')
            gradA = L.backward(gradA)
            #print(grad.shape, 'after grad')
            L.update()
        for L in self.LL0[::-1]:
            #print(grad.shape, 'before grad')
            gradV = L.backward(gradV)
            #print(grad.shape, 'after grad')
            L.update()




'''
### Xavier init: 
neurons = 256
inputs = 80*80
a = np.random.randn(neurons, inputs) / np.sqrt(inputs)
print('aaaa',a.shape)
print(a[0,:20])
#################### TESTING ###############################
import matplotlib.pyplot as plt
np.random.seed(123)
### create testdata (2D gauss)
D = np.random.multivariate_normal([-1,-1],np.diag([.4,.3]),size=100)
D_ = np.random.multivariate_normal([0,0,],np.diag([.2,.4]),size=100)
D__ = np.random.multivariate_normal([-1,1,],np.diag([.1,.5]),size=100)
D___ = np.random.multivariate_normal([1,-1],np.diag([.2,.2]),size=100)
D = np.concatenate((D,D_,D__,D___),0)
#D = D + np.array([1,1])
T = np.ones(D.shape[0])
T[:100] *= 0
T[200:300] *= 2
T[300:] *= 3
T1 = one_hot(T[:,True].T)

### build model of layers



#print('T1 / D ',T1.shape,D.shape)
INPUT_SIZE = D.shape[1]
OUTPUT_SIZE = T1.shape[1]

######## MLP ###########
A1 = layer(INPUT_SIZE,512,no_bias=False)
A2 = layer(512,256)
DO = dropout(256,.1)
A3 = layer(256,128)
A4 = layer(128,64)
AOUT = layer(64,OUTPUT_SIZE)
AOUT.f = f_lgtr
model = mlp([A1,A2,DO,A3,A4,AOUT])


######### DENSE MLP ########
#L1 = layer(4,4)
#L2 = layer(8,4)
#L3 = layer(12,4)
#L4 = layer(16,4)
#OUT = layer(20,4)
#OUT.f = f_lgtr
#Llist = [L1,L2,L3,L4,OUT]
#model = dense_mlp(Llist)

###### LARGE DENSE MLP ###################
#L1 = layer(INPUT_SIZE,OUTPUT_SIZE*2)
#L2 = layer(INPUT_SIZE+OUTPUT_SIZE*2,OUTPUT_SIZE*2)
#L3 = layer(INPUT_SIZE+OUTPUT_SIZE*4,OUTPUT_SIZE*2)
#L4 = layer(INPUT_SIZE+OUTPUT_SIZE*6,OUTPUT_SIZE*2)
#L5 = layer(INPUT_SIZE+OUTPUT_SIZE*8,OUTPUT_SIZE*2)
#L6 = layer(INPUT_SIZE+OUTPUT_SIZE*10,OUTPUT_SIZE*2)
#L7 = layer(INPUT_SIZE+OUTPUT_SIZE*12,OUTPUT_SIZE*2)
#OUT = layer(INPUT_SIZE+(OUTPUT_SIZE*6),OUTPUT_SIZE)
#OUT.f = f_lgtr
#Llist = [L1,L2,L3,OUT]
#model = dense_mlp(Llist)

errorlist = []
nodrop = []
### train model
print(D.shape)
for i in range(300):
    model.train(D,T1)
    errorlist += [model.loss]
    model.Layerlist[2].training = False
    nodrop += [qef(T1,model.infer(D))]
    model.Layerlist[2].training = True
    if (i+1) % 10 == 0:
        print('MLP : ',model.loss, ' loss at ',i+1,' steps ', )

errorlist = np.array(errorlist)

plt.figure(figsize=(10,6))
plt.plot(range(errorlist.shape[0]),errorlist,c='black',label='training loss')
plt.plot(range(errorlist.shape[0]),nodrop,c='red',label='inference loss')
plt.legend()
plt.grid()
plt.show()

#------------------------------    
### show prediction (area)
model.Layerlist[2].training = False
R = np.linspace(-2, 2, 100, endpoint=True) 
A,B = np.meshgrid(R,R)
G = [] 
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        G += [[A[i][i],A[i][j]]]
G = np.array(G)
plt.figure(figsize=(10,8))
plt.scatter(D[:,0],D[:,1],c=T*255,edgecolors=None,s=20,cmap='rainbow')
plt.scatter(G[:,0],G[:,1],c=np.argmax(model.infer(G),1)*255,edgecolors=None,s=25,cmap='spectral',alpha=.3)
plt.xlim(-2,2)
plt.ylim(-2,2)
#plt.figure(figsize=(8,8))
#a = np.flip(np.argmax(model.infer(G),1).T[::-1].reshape(100,100).T*255,1)
#plt.imshow(a)
plt.show()


'''
#'''

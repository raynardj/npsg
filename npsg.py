from datetime import datetime
from .utils import *

class npsg:
    """
    npsg doc
    """
    def __init__(self,
                 m_loss="mae",
                 m_metric=["accuracy"],
                 m_opt=Adam,
                 m_lr=1e-4,
                 m_loss_weights=False,
                 gan_name="GAN_"+datetime.now().strftime("%Y%m%d_%H%M%S"),
                 batch_size=128):
        """
        npsg core class
        """
        self.gan_name = gan_name
        self.batch_size = batch_size
        self.G=False
        self.D=False
        self.M=False
        self.m_loss=m_loss
        self.m_metric=m_metric
        self.m_opt=m_opt
        self.m_lr=m_lr
        self.m_loss_weights=m_loss_weights
        self.adv_lbl_true=0 # Adversarial Label True
        self.adv_lbl_false=1

    def set_G(self,G):
        """
        Set generative model G
        :param G: Generative model, keras model
        :return:
        """
        self.G=G
        if self.D:
            self.set_M()

    def set_D(self,D):
        """
        Set discriminative model D
        :param D: Discriminative model, keras model
        :return:
        """
        self.D=D
        if self.G:
            self.set_M()

    def set_M(self):
        """
        Set the model for training G use D as loss function
        :return:
        """
        assert self.D != False and self.G != False, "Set generative and discriminative models first"
        self.M=Model(self.G.input,self.D(self.G.output),name=self.gan_name+"_M")
        self.M_compile()

    def M_compile(self):
        self.M.compile(loss=self.m_loss,optimizer=self.m_opt(self.m_lr),metrics=self.m_metric)
        if self.m_loss_weights:
            self.M.loss_weights=self.m_loss_weights

    def train_G(self,x,y):
        """
        Training Generator
        :return: Metrics Dictionary
        """
        self._trainable(self.D,False)
        return dict(zip(self.M.metrics_names,self.M.train_on_batch(x,y,batch_size=self.batch_size)))

    def train_D(self,x,y):
        """
        Training Discriminator
        :param x: input x
        :param y: label y
        :return: Metrics Dictionary
        """
        self._trainable(self.D,True)
        return dict(zip(self.M.metrics_names,self.D.train_on_batch(x,y,batch_size=self.batch_size)))

    def _trainable(self,model,_t):
        """
        make model trainable or untrainable
        :model: the keras model
        :param d_t: True or False
        :return:
        """
        model.trainable = _t
        for l in model.layers: l.trainable = _t

class data4D:
    def __init__(self,G,real_gen,bs=128):
        """
        Historic fake data
        :param G: Generative model(keras model)
        :param real_gen: real data generator
        :param bs: batch_size, have to be even numbers
        """
        self.G=G
        self.real_gen=real_gen # generator for real data
        self.bs=bs # batch size
        assert self.bs%2 == 0, "arg:bs has to be even number"
        self.sz=int(self.bs/2)
        self.real_gen.batch_size=self.sz
        self.step=0
        self.spin=0

    def noi_func(self,noi_func):
        """
        Adapting a noise function
        :param noi_func: The noise function for generator
        :return: Nothing
        """
        self.noi=noi_func

    def fake(self):
        if self.step==0:
            self.fkdt=np.zeros(shape=tuple([self.sz]+list(self.G.output_shape)))
        self.fkdt[self.spin]=self.G.predict(self.noi(self.sz))
        if self.step<self.sz:
            fk=self.fkdt[self.spin]
        else:
            fk=self.fkdt[:,int(self.sz*np.random.rand())]
        return fk

    def march(self):
        self.step += 1
        self.spin += 1
        if self.spin >= self.sz: self.spin -= self.sz

    def __next__(self):
        f=self.fake()
        r=self.real_gen.next()
        flbl=np.ones(self.sz)
        rlbl=np.zeros(self.sz)
        self.march()
        return np.concatenate([f,r],axis=0),np.concatenate([flbl,rlbl],axis=0)





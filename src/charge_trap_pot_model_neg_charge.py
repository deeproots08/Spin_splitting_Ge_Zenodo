from config import *

#import Operators2D as Op

from Minimal_Model_2D import *


class Image_charge_calculator:

    ### Following ELECTRIC FIELDS IN DIELECTRIC MULTI-LAYERS CALCULATED BY DIGITAL COMPUTER by Takeshi Takashima an, Ryozo Ishibashi
    ### IEEE Trans. Electr. Insul, Vol EI-13, No 1, February 1978 

    ###################################################################################################################################################

    def __init__(self, q_ct, z_ct, ep_ge, ep_sige, ep_ox, t_sige, t_ox, r):
        
        self.q_ct = q_ct  # Initial charge q_1i
        self.z_ct = z_ct # position of initial charge q 
        self.ep1 = ep_ge  # dielectric constant 1 (rightmost layer)
        self.ep2 = ep_sige  # dielectric constant 2 (middle layer)
        self.ep3 = ep_ox  # dielectric constant 3 (leftmost layer, has conductor to the left of it) 
        #self.a = a  # thickness dielectric 1
        self.b = t_sige  # thickness dielectric 2
        self.c = t_ox  # thickness dielectric 3
        self.delta = r  # Small positive number for termination condition
        self.image_charges_q5s = []  # Store generated image charges
        self.image_charges_q4s = []  # Store generated image charges
        self.image_charges_q3s = []  # Store generated image charges
        self.image_charges_q2s = []  # Store generated image charges
        self.image_charges_q1s = []  # Store generated image charges

        self.alpha12 = ( self.ep1 - self.ep2 )/ (self.ep1 + self.ep2)
        self.beta12 = 1- self.alpha12

        self.alpha21 = -self.alpha12
        self.beta21 = 1 - self.alpha21

        self.alpha23 = ( self.ep2 - self.ep3 )/ (self.ep2 + self.ep3)
        self.beta23 = 1 - self.alpha23

        self.alpha32 = - self.alpha23 
        self.beta32 = 1 - self.alpha32

        #self.q0 = self.alpha12* self.q_initial
        #self.q11 = self.beta12 * self.q_initial

    def rule_i(self, q1i, z1i, level):

        if abs(q1i / self.q_ct) < self.delta:
            return  # Termination condition

        q2j = self.alpha23 * q1i
        z2j =  - z1i + 2* self.c
        q3k = self.beta23 * q1i
        z3k =  z1i
        
        self.image_charges_q2s.append((q2j, z2j, level, 'q2j'))
        self.image_charges_q3s.append((q3k, z3k, level, 'q3k'))
        
        self.rule_ii(q2j, z2j, level + 1)
        self.rule_iii(q3k, z3k, level + 1)

    def rule_ii(self, q2i, z2i, level):

        if abs(q2i / self.q_ct) < self.delta:
            return  # Termination condition
        
        q1j = self.alpha21 * q2i
        z1j = 2*(self.b + self.c) - z2i
        q5k = self.beta21 * q2i
        z5k = z2i
        
        self.image_charges_q1s.append((q1j, z1j, level, 'q1j'))
        self.image_charges_q5s.append((q5k, z5k, level, 'q5k'))

        #print("image charge = ", q5k , "co-ordinate = ", x5k, "level =", level, "\n")
        
        self.rule_i(q1j, z1j, level + 1)

    def rule_iii(self, q3i, z3i, level):

        #if abs(q3i / self.q_ct) < self.delta:
        #    return  # Termination condition

        q4j = -1 * q3i
        z4j = -z3i
        
        self.image_charges_q4s.append((q4j, z4j, level, 'q4j'))
        self.rule_iv(q4j, z4j, level + 1)

    def rule_iv(self, q4i, z4i, level):

        if abs(q4i / self.q_ct) < self.delta:
            return  # Termination condition

        q3j = self.alpha32 * q4i
        z3j = -z4i + 2*self.c
        q2k = self.beta32 * q4i
        z2k = z4i
        
        self.image_charges_q3s.append((q3j, z3j, level, 'q3j'))
        self.image_charges_q2s.append((q2k, z2k, level, 'q2k'))
        
        self.rule_iii(q3j, z3j, level + 1)
        self.rule_ii(q2k, z2k, level + 1)


    def calc_image_charges(self):

        q40 = self.q_ct
        z40 = self.z_ct
        level = 0

        q41 = -self.q_ct
        z41 = -self.z_ct
        level = 0

        self.image_charges_q4s.append((q40, z40, level, 'q4j'))
        self.image_charges_q4s.append((q41, z41, level, 'q4j'))
        self.rule_iv(q40, z40, level + 1)
        self.rule_iv(q41, z41, level + 1)

        return self.image_charges_q5s, self.image_charges_q4s, self.image_charges_q3s, self.image_charges_q2s, self.image_charges_q1s

class Image_charge_model(twoD_Minimal_Model):

    def __init__(self, n, nsm, nsc, m, ax, ay, omega=0, alphat=0, delta=0, meff_sc=1, meff_sm = 1 , mu_sc = 0, mu_sm=0, dis_mean=0, dis_var=1, px=False, py=False,  epsilon=0, ep_ge=16.2, ep_sige=12.75, ep_ox=8, t_sige=40, t_ox=8, r=1e-5 ):  # Additional arguments  in Child

        super().__init__(n, nsm, nsc, m, ax, ay, omega, alphat, delta, meff_sc, meff_sm, mu_sc, mu_sm, dis_mean, dis_var, px, py,  epsilon)  
        
        self.ep_ge = ep_ge  # dielectric constant 1 (rightmost layer)
        self.ep_sige = ep_sige  # dielectric constant 2 (middle layer)
        self.ep_ox = ep_ox  # dielectric constant 3 (leftmost layer, has conductor to the left of it) 
        #self.a = a  # thickness dielectric 1
        self.t_sige = t_sige  # thickness dielectric 2
        self.t_ox = t_ox  # thickness dielectric 3
        self.r = r  # Small positive number for termination condition

    def charge_trap_pot(self, x_ct= 0, y_ct=0, z_ct=2, z_im=0, x=0, y=0, z=0):

        ### here z_im is redundant #####

        ###q_ct = 1 because charge value is included in cont k0_e #### 

        self.image_charges_q5s, self.image_charges_q4s, self.image_charges_q3s, self.image_charges_q2s, self.image_charges_q1s = Image_charge_calculator(q_ct=-1, z_ct=z_ct, ep_ge=self.ep_ge, ep_sige=self.ep_sige, ep_ox=self.ep_ox, t_sige=self.t_sige, t_ox=self.t_ox, r=self.r).calc_image_charges()

        #print(self.image_charges_q5s)

        pot = 0

        for image_chrg in self.image_charges_q5s:

            pot = pot+ image_chrg[0]/ math.sqrt((z-image_chrg[1])**2+(y-y_ct)**2+(x-x_ct)**2)


        return k0_e/self.ep_ge*pot ### k0_e is 1/4/pi/epsilon0*ee in meV-nm 


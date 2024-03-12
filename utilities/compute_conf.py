import scipy.stats as st 
import numpy as np
import matplotlib.pyplot as plt
import statistics
from math import sqrt


MCAT_conf = [67.54658222198486, 74.37888383865356, 66.30434989929199, 75.15528202056885, 63.19875717163086]
MCAT_met_conf = [73.21428656578064, 79.05844449996948, 62.45790719985962, 74.22360181808472, 60.36789417266846]

MCAT_HViT_conf =[68.1676983833313, 72.5155234336853, 62.26707696914673, 76.70807838439941, 61.33540868759155]
MCAT_HViT_vd_conf = [68.63354444503784, 73.91304969787598, 62.42235898971558, 77.01863050460815, 60.86956262588501]
MCAT_HViT_met_conf = [74.02597665786743, 78.08440923690796, 61.78451776504517, 76.24223232269287, 60.03344655036926]
MCAT_HViT_met_vd_conf = [70.77921628952026, 79.38312292098999, 60.43771505355835, 78.57142686843872, 60.03344058990479]

MCAT_ViT_conf=[62.26707696914673, 73.29192757606506, 60.86956262588501, 74.84471797943115, 59.00620222091675]
MCAT_ViT_vd_conf=[62.26707696914673, 73.29192757606506, 60.86956262588501, 74.72826242446899, 58.85093212127686]
MCAT_ViT_met_conf =[67.53246784210205, 80.84415197372437, 66.83502197265625, 73.13664555549622, 62.54180669784546]
MCAT_ViT_met_vd_conf=[67.53246784210205, 80.84415197372437, 66.83501601219177, 73.13664555549622, 62.3745858669281]

SNN_conf=[73.29193353652954, 77.173912525177, 67.54658222198486, 78.10559272766113, 60.40372848510742]
SNN_met_conf=[65.74674844741821, 74.02597069740295, 71.04377150535583, 73.91303777694702, 66.55519008636475]

#MCAT
MCAT_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_conf)-1, 
              loc=np.mean(MCAT_conf),  
              scale=st.sem(MCAT_conf))


MCAT_met_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_met_conf)-1, 
              loc=np.mean(MCAT_met_conf),  
              scale=st.sem(MCAT_met_conf)) 

#MCAT_HViT
MCAT_HViT_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_HViT_conf)-1, 
              loc=np.mean(MCAT_HViT_conf),  
              scale=st.sem(MCAT_HViT_conf)) 


MCAT_HViT_vd_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_HViT_vd_conf)-1, 
              loc=np.mean(MCAT_HViT_vd_conf),  
              scale=st.sem(MCAT_HViT_vd_conf)) 


MCAT_HViT_met_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_HViT_met_conf)-1, 
              loc=np.mean(MCAT_HViT_met_conf),  
              scale=st.sem(MCAT_HViT_met_conf)) 


MCAT_HViT_met_vd_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_HViT_met_vd_conf)-1, 
              loc=np.mean(MCAT_HViT_met_vd_conf),  
              scale=st.sem(MCAT_HViT_met_vd_conf)) 

#MCAT-ViT
MCAT_ViT_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_ViT_conf)-1, 
              loc=np.mean(MCAT_ViT_conf),  
              scale=st.sem(MCAT_ViT_conf)) 


MCAT_ViT_vd_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_ViT_vd_conf)-1, 
              loc=np.mean(MCAT_ViT_vd_conf),  
              scale=st.sem(MCAT_ViT_vd_conf)) 


MCAT_ViT_met_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_ViT_met_conf)-1, 
              loc=np.mean(MCAT_ViT_met_conf),  
              scale=st.sem(MCAT_ViT_met_conf)) 


MCAT_ViT_met_vd_int = st.t.interval(confidence=0.99, 
              df=len(MCAT_ViT_met_vd_conf)-1, 
              loc=np.mean(MCAT_ViT_met_vd_conf),  
              scale=st.sem(MCAT_ViT_met_vd_conf))  

#SNN
SNN_int = st.t.interval(confidence=0.99, 
              df=len(SNN_conf)-1, 
              loc=np.mean(SNN_conf),  
              scale=st.sem(SNN_conf))  


SNN_met_int = st.t.interval(confidence=0.99, 
              df=len(SNN_met_conf)-1, 
              loc=np.mean(SNN_met_conf),  
              scale=st.sem(SNN_met_conf))  



def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval


plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['MCAT', 'MCAT_met', 'MCAT_HViT', 'MCAT_HViT_vd', 'MCAT_HViT_met', 'MCAT_HViT_met_vd', 'MCAT_ViT', 'MCAT_ViT_vd', 'MCAT_ViT_met', 'MCAT_ViT_met_vd', 'SNN', 'SNN_met'])
plt.title('Confidence Interval')
plot_confidence_interval(1, MCAT_conf)
plot_confidence_interval(2, MCAT_met_conf)

plot_confidence_interval(3, MCAT_HViT_conf)
plot_confidence_interval(4, MCAT_HViT_vd_conf)
plot_confidence_interval(5, MCAT_HViT_met_conf)
plot_confidence_interval(6, MCAT_HViT_met_vd_conf)

plot_confidence_interval(7, MCAT_ViT_conf)
plot_confidence_interval(8, MCAT_ViT_vd_conf)
plot_confidence_interval(9, MCAT_ViT_met_conf)
plot_confidence_interval(10, MCAT_ViT_met_vd_conf)

plot_confidence_interval(11, SNN_conf)
plot_confidence_interval(12, SNN_met_conf)

plt.show()
#!/usr/bin/env python
# coding: utf-8

# # Intro to coffea.hist

# A histogram in Coffea is a `N-D` collection of different categories, along with bin(s)

# Let's start by importing some necessary libraries

# In[1]:


from coffea import hist
import matplotlib.pyplot as plt #plot histograms
import numpy as np
import os


# Let's first explore 1D coffea histograms

# In[2]:


hist_1D = hist.Hist("Beverage items", hist.Cat("soda", "different soda"), hist.Bin("x","x coordinate [m]",20, -5,5))


# Retrieving the histogram's bin contents using `values()`

# In[3]:


hist_1D.values()


# Let's use `fill()` to add different kinds of soda, with random x values using `numpy.random`

# In[4]:


hist_1D.fill(soda="cocacola",x=np.random.normal(size=10))


# In[5]:


hist_1D.values()


# Let's fill the hist with another soda item, again using `numpy.random`.. this time with a weight of 5 each.

# In[6]:


hist_1D.fill(soda="pepsi",x=np.random.normal(size=10), weight=np.ones(10)*5)


# In[7]:


hist_1D.values()


# Let's use `axes()` to explore the histogram

# In[8]:

hist_1D.axes()


# Let's use `integrate()`. Coffea hist documentation says `integrate()` --> Integrates current histogram along one dimension

# In[9]:


print(hist_1D.integrate('soda'))


# In[10]:


print(hist_1D.integrate('soda').values())


# In[11]:


print(hist_1D.integrate('x'))


# In[12]:


print(hist_1D.integrate('x').values())


# Let's use matplotlib and coffea `plot1d` to draw a histogram

# In[13]:


hist.plot1d(hist_1D.integrate('soda'),stack=True)


# Let's move to 2D histograms. Simple example from the Coffea manual plus Brent's tutorial notebook from last year.

# In[14]:


h = hist.Hist("Observed bird count", hist.Cat("species", "Bird species"), hist.Bin("x", "x coordinate [m]", 10, -5, 5), hist.Bin("y", "y coordinate [m]", 10, -5, 5))


# Now we'ss use `fill()` to add 10 `ducks`, with random `x-y` values using `numpy.random`, each with a weight of 3

# In[15]:


h.fill(species='ducks', x=np.random.normal(size=10), y=np.random.normal(size=10), weight=np.ones(10) * 3)


# Now I'll add another species

# In[16]:


h.fill(species='phoenix', x=np.random.normal(size=8), y=np.random.normal(size=8), weight=np.ones(8)*9)


# In[17]:


h.values()


# Now let's use `integrate()` for 2d coffea histograms

# In[18]:


h.integrate('species')


# In[19]:


h.integrate('species').values()


# In[20]:


h.integrate('species','ducks').values()


# In[21]:


h.integrate('x')


# In[22]:


h.integrate('x').values()


# In[23]:


h.integrate('y').values()


# In[24]:


h.integrate('species').integrate('x').values()


# Let's create a plot to draw everthing in using matplotlib and the `plot2d()` method in `coffea.hist`

# In[25]:


hist.plot2d(h.integrate('species'), xaxis='x')


# Now we can play with the axes to learn some more<br>
# We can view the axes with `h.axes()`

# In[26]:


h.axes()


# We can remove the `x`-axis by integrating it out with `integrate()`

# In[27]:


h.integrate('x').integrate('species').values()


# And now we can make a `1D` plot, in this case of `species` and `y coordinate`

# In[28]:


hist.plot1d(h.integrate('x'), stack=True) #stack makes a stack plot


# ## A more practical, physics example

# In this example, I'll load a set of histograms from `histos/all2017mcsigsamples_skipSR_2022sept13_topcoffeatutorial.pkl.gz`<br>
# This is a pickle file created by TopCoffea

# First, let's import all the relevent packages (same as before, but here to make this section stand alone)

# In[29]:


import pickle #read pickle file
import gzip #read zipped pickle file


# Next, we'll open the pickle file, and load its histograms into a dictionary

# In[30]:

# create a dir where pkl file will be downloaded from web area on earth
pkl_dir = "tutorialpkldir"
if not os.path.exists(pkl_dir):
    os.makedirs(pkl_dir)
os.system("curl -o pkl_dir/all2017mcsigsamples_skipSR_2022sept13_topcoffeatutorial.pkl.gz https://www.crc.nd.edu/~abasnet/EFT/topcoffeaTutorial/all2017mcsigsamples_skipSR_2022sept13_topcoffeatutorial.pkl.gz")
fin = 'pkl_dir/all2017mcsigsamples_skipSR_2022sept13_topcoffeatutorial.pkl.gz'
hists = {} #dictionary of histograms
with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
        if k in hists: hists[k]+=hin[k]
        else: hists[k]=hin[k]
        print(hists[k])


# Now we'll grab the histogram for `njets`

# In[31]:


h = hists['njets'] #load histogram of njets distribution


# Each histogram is a `N-D` collection of different categories

# In[32]:


h.axes() #all axes in this version


# You can retrieve the histogram's bin contents with the `values()` method

# In[33]:


h.values() #this is large, and Jupyter wants to show the whole thing
print(list(h.values())[0],'...') #just print the first entry


# Using `identifiers()`, you can list different categories inside an axis.

# In[34]:


h.axis('appl').identifiers()


# In[35]:


h.axis('sample').identifiers()


# In[36]:


h.axis('channel').identifiers()


# To select a specific label in a category we must use `integrate()` (the other option is `sum()` which combines all the lables in a category together)

# In[37]:


h = h.integrate('channel','2los_CRZ').integrate('systematic', 'nominal').integrate('appl','isSR_2lOS')


# We've now integrated outeverything but the type of samples:

# In[38]:


h.values()


# In[39]:


h.axes()


# Let's create a plot to draw everthing in using matplotlib and the `plot1d()` method in `coffea.hist`

# In[40]:


fig, ax = plt.subplots(1,1, figsize=(7,7)) #create an axis for plotting
hist.plot1d(h, stack=True)
#fig.show() #not needed in Jupyter, but this draws the figure in the terminal


# # topcoffea.modules.HistEFT

# I'll continue using the pkl file from above<br>
# Now we'll use methods that are unique to HistEFT (e.g. `set_wilson_coefficients()` to scale the Wilson Coefficient (WC) values)

# The `HistEFT` class holds the structure constants ($S_0, S_{1j}, S_{2j},$ and $S_{3jk}$) we solved for when partins the EFT files, <br>
# so the event yields are just a function of the WCs ($\vec{c}$):

# \begin{equation}
# N\left(\dfrac{\vec{c}}{\Lambda^2}\right) = S_0 + \sum_j S_{1j} \frac{c_j}{\Lambda^2} + \sum_j S_{2j} \frac{c_j^2}{\Lambda^4} + \sum_{j,k} S_{3jk} \frac{c_j}{\Lambda^2} \frac{c_k}{\Lambda^2}
# \end{equation}

# In[41]:


h._nwc


# where `_nwc` is a local variable inside a HistEFT that stores how many WCs it contains<br>

# In[42]:


h._wcnames


# First, we'll scale the histogram to the SM (all `WCs=0`)

# In[43]:


h.set_sm()


# The WCs are used whenever `values()` method is called

# In[44]:


h.values()


# In[58]:


h.values(overflow='allnan')


# Plotting this should look the same as before, since by default the WCs are 0

# In[46]:


hist.plot1d(h, stack=True)


# Now let's set some of the WCs to 1 to see that things change

# In[47]:


h.set_wilson_coefficients(cpQM=1, ctW=1, ctG=1)


# In[48]:


h.values()


# In[49]:


hist.plot1d(h, stack=True)


# There's one last thing we must due in order to produce the predicted event yields.<br>
# The EFT samples come normalized to $\sigma * w_{\mathrm{gen}}$<br>
# In order to produce event yields, we must scale them by $\frac{\mathcal{L}}{\sum{w_{\mathrm{event}}^{\mathrm{SM}}}}$ , where $\sum{w_{\mathrm{event}}^{\mathrm{SM}}}$ is the sum of the event weights, evaluated at the SM. When we run the topcoffea processor to make our pkl file, this scaling with $\sum{w_{\mathrm{event}}^{\mathrm{SM}}}$ is already done. We just need to scale by lumi.

# In[50]:


h.set_sm()    #set WCs to be SM values to be safe
wgt = 1000*41.48 #multiply 1000 for units
print('Scaling by', wgt)
h.scale(wgt) #2017 lumi of 41.48 fb^-1


# In[55]:


import mplhep as hep
hep.style.use("CMS")

fig, ax = plt.subplots(1,1) #create an axis for plotting
hist.plot1d(h, ax=ax, stack=True)
ax.legend()

# add some labels
lumi = hep.cms.label(ax=ax, lumi='41.48', label="Preliminary")


# In[ ]:





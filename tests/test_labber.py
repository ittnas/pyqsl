import numpy as np
import sys
sys.path.insert(0, '/usr/share/Labber/Script') # For Labber
import Labber
#sys.path.insert(0,'/usr/share/Labber/Script/Labber/_include36')
#import InstrumentConfig
#lLog = [dict(name='y'),dict(name='w'),dict(name='xb')]
#lLog = [dict(name='time'),dict(name='freq'),dict(name='freq_v'),dict(name='time_v')]
#lLog = [dict(name='time_v',x_name='time'),dict(name='freq_v',x_name='freq')]
#lLog = [dict(name='time_v',x_name='time')]
lLog = [dict(name='time_v',x_name='time'),dict(name='freq_v',x_name='freq')]

x = [1.0,1.2]
#step_channel = [dict(name='x',values=x),dict(name='p1',values=[1.0,1.2])]
step_channel = [dict(name='x',values=x)]
l = Labber.createLogFile_ForData('TestLog', lLog,step_channel)
#l = Labber.createLogFile_ForData('TestLog', lLog)
time = np.linspace(0,5,21)
freq = np.linspace(0,2,31)

time_v = 2*time
freq_v = freq*3
#d = _InstrumentQuantity.getTraceDict(time_v,x=time)
#d = InstrumentConfig.InstrumentQuantity.getTraceDict(time_v,x=time)
dt = Labber.getTraceDict(time_v,x=time)
df = Labber.getTraceDict(freq_v,x=freq)
print(dt)
#d = Labber.LogFile._Labber.LogFile__getTraceDict(time_v,x=time)
#l.addEntry({"time":time,"time_v":time_v,"freq":freq,"freq_v":freq_v})
#print(d)

#l.addEntry([{"time_v":dt}])
#l.addEntry([{"time_v5":dt}])
l.addEntry({'time_v':dt,'freq_v':df})
#l.addEntry({'time_v':time_v,'time':time,'freq_v':freq_v,'freq':freq})
#l.addEntry({'time_v':time_v,'time':time,'freq_v':freq_v,'freq':freq})
#l.addEntry({'time_v':time_v,'time':time})
#l.addEntry([{"time_v":dt}])
#l.addEntry([{"time_v":d},{"freq_v":d}])
#.addEntry([d])


#def getTraceDict(value=[], x0=0.0, dx=1.0, bCopy=False,
#                 x1=None, x=None, logX=False, t0=None, dt=None):
#l.addEntry({"x":1.0,'y':np.array([1.0])})
#l.addEntry({"y":dict(name='y2',values=np.array([1, 2, 3]))})
#l.addEntry({"x":2.0,'y':np.array([2.0])})
# x2 = np.linspace(0,2,51)
# #data = {'y': np.sin(2*np.pi*5*x2),'xb':xb,'w':np.sin(2*np.pi*5*xb)}
# #data = {'y': {"x2":x2,"y2":np.sin(2*np.pi*5*x2)}}
# data = {'y':np.sin(2*np.pi*5*x2)}
# l.addEntry(data)
# x3 = np.linspace(0,2,101)
# data = {'y': np.sin(2*np.pi*5*x2)}
# l.addEntry(data)

#help(Labber.getTraceDict)

"""
Created 21. January 2020 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy
import os
import sys
import tmm
import glob
import lmfit
import pickle
import matplotlib.pyplot as plt
import quantities as pq
from quantities import UncertainQuantity as uq

os.system('mv result.log result.alt')
sys.stdout=open('result.log','a')

#Funktionen
def conv(t):
	return t.replace(',','.')

def Sellmeier(lambdas,A,B1,C1):
	return numpy.sqrt(A+B1*lambdas/(lambdas-C1))

def TM(params):
	prm=params.valuesdict()
	global Tsim
	Tsim=numpy.array([])
	global Rsim
	Rsim=numpy.array([])
	global Ssim
	Ssim=prm['Cs']/lamb**prm['alpha']
	global n_collect
	n_collect=numpy.array([])
	global k_collect
	k_collect=numpy.array([])
	d_list=[numpy.inf,prm['d'],numpy.inf]
	c_list=['i' for i in range(len(d_list))]
	for i,l in enumerate(lamb):
		n=Sellmeier(l,prm['A'],prm['B1'],prm['C1'])
		k=epsilon[numpy.where(lamb_epsilon==l)][0]*l/(4*numpy.pi)*prm['Cd']
		n_collect=numpy.append(n_collect,n)
		k_collect=numpy.append(k_collect,k)
		if numpy.isnan(n)==True:
			return 1e99
			break
		nk_list=[1,n+k*1j,1]
		sim=tmm.inc_tmm('s',nk_list,d_list,c_list,0+0j,l)
		Tsim=numpy.append(Tsim,sim['T'])
		Rsim=numpy.append(Rsim,sim['R'])
	global res
	res=Tmeas-(Tsim-Ssim)
	global Rquadrat
	Rquadrat=1-numpy.var(res)/numpy.var(Tmeas)
	# ~ print(Rquadrat)
	return res

#Einlesen
files=glob.glob('*_c01_*.txt')

filenames_collect=[]
d_collect=[]
A_collect=[]
B1_collect=[]
C1_collect=[]
Cd_collect=[]
Cs_collect=[]
alpha_collect=[]

for i in files:
	filename=os.path.splitext(i)[0]
	print(filename)
	lamb0,Tmeas0=numpy.genfromtxt((conv(t) for t in open(i)),unpack=True,delimiter='\t',skip_header=36,skip_footer=2)
	Tmeas0=Tmeas0/100
	args=numpy.where((lamb0>=400)&(lamb0<=900))
	argsred=numpy.where((lamb0>=400)&(lamb0<=900))[0][::25]

	#Attenuationskoeffizienten berechnen
	lambref,Ileer=numpy.genfromtxt((conv(t) for t in open('Counts_DMAcLiCl_ohne_Cellulose_c01_000000.txt')),unpack=True,delimiter='\t',skip_header=36,skip_footer=2)
	lamb_epsilon,Ivoll=numpy.genfromtxt((conv(t) for t in open('Counts_Cellulose_DMAcLiCl_c01_000001.txt')),unpack=True,delimiter='\t',skip_header=36,skip_footer=2)
	cKuevette=3.2e-4
	dKuevette=1e-2
	A=numpy.log10(Ileer/Ivoll)
	epsilon=A/(cKuevette*dKuevette)

	#Anpassen
	params=lmfit.Parameters()
	params.add('d',1e5,vary=False)
	params.add('A',1,min=0.5,max=3)
	params.add('B1',1,min=0.05,max=3)
	params.add('C1',100,min=0,max=400)
	params.add('Cd',1,min=0,max=1)
	params.add('Cs',1,min=0,max=1000)
	params.add('alpha',1,min=0,max=2)

	#Vorgeschaltete globale Loesungssuche
	lamb=lamb0[argsred]
	Tmeas=Tmeas0[argsred]
	result0=lmfit.minimize(TM,params,method='differential_evolution')

	params=lmfit.Parameters()
	prm0=result0.params.valuesdict()
	params.add('d',prm0['d'],vary=False)
	params.add('A',prm0['A'])
	params.add('B1',prm0['B1'])
	params.add('C1',prm0['C1'])
	params.add('Cd',prm0['Cd'])
	params.add('Cs',prm0['Cs'])
	params.add('alpha',prm0['alpha'])
	#Bis hierher ausschalten

	lamb=lamb0[args]
	Tmeas=Tmeas0[args]
	result=lmfit.minimize(TM,params,method='least_squares')

	print('R^2:',Rquadrat)
	result.params.pretty_print()

	#Speichern und laden
	pickle.dump(result,open(str(filename)+'_result','wb'))
	result=pickle.load(open(str(filename)+'_result','rb'))

	#Sammeln
	prm=result.params.valuesdict()
	err={}
	for key in result.params:
		err[key]=result.params[key].stderr
	filenames_collect.append(filename)
	d_collect.append(uq(prm['d'],pq.nm,err['d']))
	A_collect.append(uq(prm['A'],pq.dimensionless,err['A']))
	B1_collect.append(uq(prm['B1'],pq.dimensionless,err['B1']))
	C1_collect.append(uq(prm['C1'],pq.nm,err['C1']))
	Cd_collect.append(uq(prm['Cd'],pq.dimensionless,err['Cd']))
	Cs_collect.append(uq(prm['Cs'],pq.dimensionless,err['Cs']))
	alpha_collect.append(uq(prm['alpha'],pq.dimensionless,err['alpha']))

	#Auftragen
	params=lmfit.Parameters()
	for key,val in dict(result.params.valuesdict()).items():
		params.add(key,val,vary=False)

	lamb=lamb0[args]
	Tmeas=Tmeas0[args]
	TM(params)

	plt.clf()
	plt.plot(lamb,Tmeas,'k')
	plt.plot(lamb,Tsim)
	plt.plot(lamb,Tsim-Ssim)
	plt.plot(lamb,1-(Tsim-Ssim)-Rsim)
	plt.plot(lamb,Rsim)
	plt.plot(lamb,Ssim)
	plt.savefig(filename+'_disp.pdf')
	plt.savefig(filename+'_disp.png')

	fig,ax1=plt.subplots()
	ax2=ax1.twinx()
	ax1.plot(lamb,n_collect,'k')
	ax2.plot(lamb,k_collect)
	plt.savefig(filename+'_nk_collect.pdf')
	plt.savefig(filename+'_nk_collect.png')

	#Exportieren
	export=numpy.array([lamb,Tmeas,Tsim,Ssim,n_collect,k_collect])
	numpy.savetxt(filename+'.tsv',export.transpose(),delimiter='\t',newline='\n')

f=open('Fitparameter.csv','w')
f.write('Dateiname,d_collect+-unc,A_collect+-unc,B1_collect+-unc,C1_collect+-unc,c_collect+-unc,alpha_collect+-unc\n')
for i,filename in enumerate(filenames_collect):
	f.write(filename)
	for j in [d_collect[i],A_collect[i],B1_collect[i],C1_collect[i],Cd_collect[i],Cs_collect[i],alpha_collect[i]]:
		f.write(','+str(float(j.magnitude))+'+-'+str(float(j.uncertainty)))
	f.write('\n')

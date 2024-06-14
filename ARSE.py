"""
Created 14. June 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import tmm
import glob
import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from quantities import UncertainQuantity as uq

def Sellmeier(lambdas,A,B1,C1):
	return np.sqrt(A+B1*lambdas/(lambdas-C1))

def TM(params):
	prm=params.valuesdict()
	global Tsim;Tsim=np.array([])
	global Rsim;Rsim=np.array([])
	global Ssim;Ssim=prm['Cs']/lamb**prm['alpha']
	global n_collect;n_collect=np.array([])
	global k_collect;k_collect=np.array([])
	d_list=[np.inf,prm['d'],np.inf]
	c_list=['i' for i in range(len(d_list))]
	for i,l in enumerate(lamb):
		n=Sellmeier(l,prm['A'],prm['B1'],prm['C1'])
		k=epsilon[np.where(lamb_epsilon==l)][0]*l/(4*np.pi)*prm['Cd']
		n_collect=np.append(n_collect,n)
		k_collect=np.append(k_collect,k)
		if np.isnan(n):
			return 1e99
			break
		nk_list=[1,n+k*1j,1]
		sim=tmm.inc_tmm('s',nk_list,d_list,c_list,0+0j,l)
		Tsim=np.append(Tsim,sim['T'])
		Rsim=np.append(Rsim,sim['R'])
	return Tmeas-(Tsim-Ssim)

for i in glob.glob('*_c01_*.txt'):
	filename=os.path.splitext(i)[0]
	lamb0,Tmeas0=np.genfromtxt((t.replace(',','.') for t in open(i)),unpack=True,delimiter='\t',skip_header=36,skip_footer=2)
	Tmeas0=Tmeas0/100
	args=np.where((lamb0>=400)&(lamb0<=900))
	lamb=lamb0[args];Tmeas=Tmeas0[args]

	lambref,Ileer=np.genfromtxt((t.replace(',','.') for t in open('Counts_DMAcLiCl_ohne_Cellulose_c01_000000.txt')),unpack=True,delimiter='\t',skip_header=36,skip_footer=2)
	lamb_epsilon,Ivoll=np.genfromtxt((t.replace(',','.') for t in open('Counts_Cellulose_DMAcLiCl_c01_000001.txt')),unpack=True,delimiter='\t',skip_header=36,skip_footer=2)
	cKuevette=3.2e-4
	dKuevette=1e-2
	A=np.log10(Ileer/Ivoll)
	epsilon=A/(cKuevette*dKuevette)

	params=lm.Parameters()
	params.add('d',1e5,vary=False)
	params.add('A',1,min=0.5,max=3)
	params.add('B1',1,min=0.05,max=3)
	params.add('C1',100,min=0,max=400)
	params.add('Cd',1,min=0,max=1)
	params.add('Cs',1,min=0,max=1000)
	params.add('alpha',1,min=0,max=2)

	result=lm.minimize(TM,params,method='least_squares')
	results.params.pretty_print()
	for key in results.params:
		print([key,results.params[key].value,results.params[key].stderr],file=open(filename+'_params.txt','a'))

	TM(result.params)

	plt.close('all')
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

	np.savetxt(filename+'.dat',np.array([lamb,Tmeas,Tsim,Ssim,n_collect,k_collect]).transpose())



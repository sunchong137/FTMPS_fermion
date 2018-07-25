#this is a python script that calls a c++ excutable
import time
import numpy as np
import subprocess
import os

def call_ftmps(norb, nimp, u, mu, beta, tau, h1e, maxm=2000, \
               tmpdir='./', mpsdir='../'):
    tmpdir = "/home/sunchong/work/finiteTMPS/tests"
    time_str = repr(time.time())[6:]
    hamfile = tmpdir + "/" + time_str + "hamfile.txt"
    infile = tmpdir + "/" + time_str + "input_mps"
    impsite = tmpdir + "/" + time_str + "impsite.txt"

    # write input file
    fin = open(infile, "w")
    fin.write("input\n{\n")
    fin.write("hamfile = %s\n"%hamfile)
    fin.write("outdir = %s\n"%(tmpdir+time_str))
    fin.write("impsite = %s\n"%(impsite))
    fin.write("N = %d\n"%norb)
    fin.write("Nimp = %d\n"%nimp)
    fin.write("U = %f\n"%u)
    fin.write("mu = %f\n"%mu)
    fin.write("beta = %f\n"%beta)
    fin.write("tau = %f\n"%tau)
    fin.write("maxm = %d\n"%maxm)
    fin.write("cutoff = 1E-11\n")
    fin.write("realstep = false\n")
    fin.write("verbose = no\n}")
    fin.write("fitmpo = yes\n")
    fin.write("rungekutta = no\n")
    fin.close()

    # write 1 body hamiltonian 
    h1e_n = h1e.reshape(2*norb, norb)
    np.savetxt(hamfile,h1e_n)
    
    np.savetxt(impsite, (np.arange(nimp)+1).T, fmt="%d")

    # call ancilla code
    subprocess.call([mpsdir + "impsolver_ancilla", infile])

    rdm1 = np.loadtxt(tmpdir+time_str+"rdm1s.txt")
    e = np.loadtxt(tmpdir+time_str+"energy.txt")
    # read rdms
    #e = np.loadtxt(tmpdir+time_str+"energy.txt")
    #rdm1 = np.loadtxt(tmpdir+time_str+"rdm1s.txt")
    #rdm2 = np.loadtxt(tmpdir+time_str+"rdm2.txt")
    #print e
    #print rdm1
    #print rdm2
    
    # remove temperary files
    os.remove(infile)
    os.remove(hamfile)
    os.remove(impsite)
    os.remove(tmpdir+time_str+"energy.txt")
    os.remove(tmpdir+time_str+"rdm1s.txt")
    os.remove(tmpdir+time_str+"rdm2.txt")
    return e, rdm1
    
    
def printmat(m):
    nx, ny = m.shape
    for i in range(nx):
        print '   '.join(map(str, m[i]))


if __name__ == "__main__":
    from pyscf.ftsolver import ed_grandmu as ftfci

    #norb = 4
    #nelec = 4
    #nimp = 2
    u = 4.
    mu = u/2
    beta = 0.4
    T = 1./beta
    tau = 0.1
    h1e = np.loadtxt("hamfile.txt")
    norb = h1e.shape[0]/2
    h1ea = h1e[:norb]
    h1eb = h1e[norb:]
    nimp = 4
    nelec = norb
    #h1ea = np.zeros((norb, norb))
    #for i in range(norb):
    #    h1ea[i,(i+1)%norb] = -1
    #    h1ea[i,(i-1)%norb] = -1.
    #h1ea[3,3] = -0.5
    #h1ea[3,1] = -0.5
    #h1ea[0,-1] = 0
    #h1ea[-1,0] = 0
    #h1eb = h1ea.copy()
    g2e = np.zeros((norb,norb,norb,norb))

    #print "Hamiltonian:\n"
    #printmat(h1ea)    

    for i in range(nimp):
        g2e[i,i,i,i] = u
    e, rdm1 = call_ftmps(norb, nimp, u, mu, beta, tau, np.array([h1ea, h1eb])) 
    #for i in range(norb):
    #    h1ea[i,i] -= 1*mu
    #    h1eb[i,i] -= 1*mu
    #dm0, _, e0 = ftfci.rdm12s_fted((h1ea, h1eb), (g2e*0, g2e, g2e*0),norb,nelec,T,mu=0, symm='UHF')
    #
    #print "------------------ed result-------------------- :\n"
    #print "energy = ", e0
    #print
    #print "rdm1 = \n", dm0
    #print
    #print "------------------mps result------------------- :\n"
    #print "energy = ", e
    #print
    #print "rdm1 = \n", rdm1
    #
    #print "energy difference: ", (e-e0)/norb
    #print "rdm difference: ", np.linalg.norm(rdm1[:norb]-dm0[0])/norb

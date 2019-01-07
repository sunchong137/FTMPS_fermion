#this is a python script that calls a c++ excutable
import time
import numpy as np
import subprocess
import os

def generate_embedding(norb,nimp,u,afm):
    '''
    Generate the 1-body embedding Hamiltonian.
    '''

    tmpdir = "/home/sunchong/work/finiteTMPS/tests/call/dump/"
    impsite = np.arange(nimp)+1
    np.savetxt(tmpdir+"/impsite.txt",impsite.T,fmt='%d')

    # solve full system
    nelec = norb
    nemb = 2*nimp
    h1e  = np.zeros((2, norb, norb))
    for i in range(norb):
        h1e[0][i,(i+1)%norb] = -1.
        h1e[0][i,(i-1)%norb] = -1.
        h1e[1][i,(i+1)%norb] = -1.
        h1e[1][i,(i-1)%norb] = -1.
        if(i%2==0):
            h1e[0][i,i] = afm
            h1e[1][i,i] = -afm
        else:
            h1e[1][i,i] = afm
            h1e[0][i,i] = -afm

    ewa, eva = np.linalg.eigh(h1e[0])
    ewb, evb = np.linalg.eigh(h1e[1])
    dm1a = np.einsum('ij,kj-> ik', eva[:,:nelec/2],eva[:,:nelec/2].conj())
    dm1b = np.einsum('ij,kj-> ik', evb[:,:nelec/2],evb[:,:nelec/2].conj())
    
    # construct bath
    Ra = np.zeros((norb,nimp*2))
    Rb = np.zeros((norb,nimp*2))
    Ra[:nimp,:nimp] = np.eye(nimp)
    Rb[:nimp,:nimp] = np.eye(nimp)
    _,_,ba = np.linalg.svd(dm1a[:nimp,nimp:],full_matrices=False)
    _,_,bb = np.linalg.svd(dm1b[:nimp,nimp:],full_matrices=False)
    Ra[nimp:,nimp:] = ba.conj().T
    Rb[nimp:,nimp:] = bb.conj().T
    
    # construct 1-body embedding Hamiltonian
    h1emb = np.zeros((2, 2*nimp,2*nimp))
    h1emb[0] = np.dot(Ra.conj().T, np.dot(h1e[0], Ra))
    h1emb[1] = np.dot(Rb.conj().T, np.dot(h1e[1], Rb))
    h1emb    = h1emb.reshape(nemb*2,nemb)
    np.savetxt(tmpdir+"/hamfile.txt",h1emb)
    np.savetxt(tmpdir+"/ehamfile.txt",h1emb)

    #construct 2-body embedding Hamiltonian
    g2emb = np.einsum('ki,im,li,in -> kmln', Ra.T.conj(),Ra,Rb.T.conj(),Rb)*u
    np.savetxt(tmpdir+"/evfile.txt", g2emb.reshape(nemb**3,nemb))

    #g2e = np.zeros((nemb,)*4) 
    #for i in range(nimp):
    #    g2e[i,i,i,i] = u
    #g2e[1,1,2,2] = 2.
    #g2e[2,2,1,1] = 2.
    #g2emb = g2e.copy()
    #g2e = g2e.reshape(nemb**3,nemb)
    #np.savetxt(tmpdir+"/vfile.txt",g2e)

    return (h1emb[:nemb], h1emb[nemb:]), (g2emb*0, g2emb,g2emb*0)

    #############################################
    ##Hubbard Hamiltonian
    #nemb = 2*nimp
    #h1a = np.zeros((nemb,nemb))
    #for i in range(nemb):
    #    h1a[i,(i+1)%nemb] = -1.
    #    h1a[i,(i-1)%nemb] = -1.
    #h1 = np.array([h1a,h1a])
    #np.savetxt(tmpdir+"/hamfile.txt",h1.reshape(2*nemb,nemb))
    #return (h1a,h1a)
    

    #############################################




def call_ftmps(norb, nimp, u, mu, beta, tau, maxm=2000, \
               tmpdir='./', mpsdir='../../'):

    '''
    Solve embedding problem with impsolver_ancilla
    '''

    tmpdir = "/home/sunchong/work/finiteTMPS/tests/call/dump/"
    #time_str = repr(time.time())[6:]
    hamfile = tmpdir + "/hamfile.txt"
    ehamfile = tmpdir + "/ehamfile.txt"
    vfile   = tmpdir + "/vfile.txt"
    evfile   = tmpdir + "/evfile.txt"
    infile = tmpdir + "/input_mps"
    impsite = tmpdir + "/impsite.txt"

    # write input file
    fin = open(infile, "w")
    fin.write("input\n{\n")
    fin.write("hamfile = %s\n"%hamfile)
    fin.write("ehamfile = %s\n"%ehamfile)
    fin.write("vfile = %s\n"%vfile)
    fin.write("evfile = %s\n"%evfile)
    fin.write("outdir = %s\n"%(tmpdir))
    fin.write("impsite = %s\n"%(impsite))
    fin.write("N = %d\n"%norb)
    fin.write("Nimp = %d\n"%nimp)
    fin.write("U = %f\n"%u)
    fin.write("mu = %f\n"%mu)
    fin.write("beta = %f\n"%beta)
    fin.write("tau = %f\n"%tau)
    fin.write("maxm = %d\n"%maxm)
    fin.write("cutoff = 1E-9\n")
    fin.write("realstep = yes\n")
    fin.write("verbose = no\n")
    fin.write("fitmpo = yes\n")
    fin.write("rungekutta = yes\n}")
    fin.close()

    # write 1 body hamiltonian 
    #h1e_n = h1e.reshape(2*norb, norb)
    #np.savetxt(hamfile,h1e_n)
    
    #np.savetxt(impsite, (np.arange(nimp)+1).T, fmt="%d")

    # call ancilla code
    subprocess.call([mpsdir + "impsolver_ancilla_ibath", infile])

    rdm1 = np.loadtxt(tmpdir+"/rdm1s.txt")
    e = np.loadtxt(tmpdir+"/energy.txt")
    
    # remove temperary files
    os.remove(infile)
    os.remove(hamfile)
    os.remove(impsite)
    os.remove(tmpdir+"/energy.txt")
    os.remove(tmpdir+"/rdm1s.txt")
    os.remove(tmpdir+"/rdm2.txt")
    return e, rdm1
    
    
def printmat(m):
    nx, ny = m.shape
    for i in range(nx):
        print '   '.join(map(str, m[i]))


if __name__ == "__main__":
    from pyscf.ftsolver import ed_grandmu as ftfci

    tmpdir = "/home/sunchong/work/finiteTMPS/tests/call/dump/"
    norb = 12
    nimp = 2
    nemb = 2*nimp
    u = 4.
    mu = 0. #u/2
    beta = 0.4
    T = 1./beta
    tau = 0.05
    afm = 0.01

    h1e, g2e = generate_embedding(norb,nimp,u,afm)
    #g2e = np.zeros((nemb,)*4)

    #for i in range(nimp):
    #    g2e[i,i,i,i] = u
    #g2e = (g2e*0,g2e,g2e*0)
    
    e, rdm1 = call_ftmps(nemb, nimp, u, mu, beta, tau) 
    rdm1fci,_,efci = ftfci.rdm12s_fted(h1e,g2e,nemb,nemb,T,mu,symm="UHF")
    rdm1fci = rdm1fci.reshape(2*nemb,nemb)
    
    print e, efci
    print np.linalg.norm(rdm1-rdm1fci)

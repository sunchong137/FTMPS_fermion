#include "itensor/all.h"
#include <iostream>
#include "TStateObserver.h"
//#include "S2.h"

using namespace std;
using namespace itensor;

using TensorT = ITensor;
using MPOT = MPOt<TensorT>;
using MPST = MPSt<TensorT>;

double** get_rdm1up(MPST psi, int N);
double** get_rdm1dn(MPST psi, int N);
double *get_rdm2diag(MPST psi, int N);
double*** get_rdm1s(MPST psi, int N);


int 
main(int argc, char* argv[])
    {
    printfln("TensorT == %s",(std::is_same<TensorT,ITensor>::value ? "ITensor" : "IQTensor"));

    //Get parameter file
    if(argc != 2)
        {
        printfln("Usage: %s inputfile.",argv[0]);
        return 0;
        }
    auto input = InputGroup(argv[1],"input");

    auto N = input.getInt("Nx",10);
    //auto Ny = input.getInt("Ny",1);
    auto Npart = input.getInt("Npart",1);
    auto U = input.getReal("U", 1.0);
    auto mu = input.getReal("mu", 1.0);
    //auto V1 = input.getReal("V1",1.0);
    auto t1 = input.getReal("t1", 1.0);
    //auto t2 = input.getReal("t2", 1.0);
    auto periodic = input.getYesNo("periodic",false);
    //auto lattice_type = input.getString("lattice_type","square");

    auto beta = input.getReal("beta",1);
    auto tau = input.getReal("tau",0.005);

    auto maxm = input.getInt("maxm",1000);
    auto cutoff = input.getReal("cutoff",1E-11);

    auto realstep = input.getYesNo("realstep",false);
    auto verbose = input.getYesNo("verbose",false);

    //auto N = Nx*Ny;

    Args args;
    args.add("N",N);
    //args.add("Ny",Ny);
    //args.add("Npart",Npart);
    args.add("Maxm",maxm);
    args.add("Cutoff",cutoff);
    //args.add("YPeriodic",periodic);
    args.add("Verbose",verbose);

    auto sites = Hubbard(2*N, {"ConserveNf",false,"ConserveSz", true});
    //auto sites = Hubbard(2*N, {"ConserveNf",true,"ConserveSz", true});
    //auto sites = Hubbard(2*N);

    
    ////////////////////////////////////////////////////////
    //               Construct Hamitltonian               //
    ////////////////////////////////////////////////////////
    auto ampo = AutoMPO(sites);

    //////////////////
    // Coulomb term //
    //////////////////
    for(int i=1; i<=N;++i ) 
    {
        int s1 = 2*i-1;
        ampo += U, "Nupdn", s1;
    }
    ///////////////////////////
    // nearest neighbor term //
    ///////////////////////////

    int endpnt; //PBC or OBC
    if(periodic) endpnt = N;
    else endpnt = N-1;
    
    for(int i=1; i<=endpnt;++i )
    {
        int s1 = 2*i-1;
        int s2 = 2*(i+1)-1;
        if(i==N) s2 = 1;
        ampo += -t1,"Cdagup",s1,"Cup",s2;
        ampo += -t1,"Cdagup",s2,"Cup",s1;
        ampo += -t1,"Cdagdn",s1,"Cdn",s2;
        ampo += -t1,"Cdagdn",s2,"Cdn",s1;
    }
    /////////////////////////////
    // chemical potential term //
    /////////////////////////////

    mu = U/2;
    for(int i=1; i<=N; ++i) 
    {
        int s1 = 2*i-1;
        ampo += -mu, "Nup", s1;
        ampo += -mu, "Ndn", s1;
    }

    auto H = MPOT(ampo);

    ///////////////////////////////////////////////////
    //             total number operator             //
    ///////////////////////////////////////////////////
    auto nmpo = AutoMPO(sites);
    for(int i=1; i<=N; ++i)
    {
        int s1 = 2*i-1;
        nmpo += 1.0, "Nup", s1;
        nmpo += 1.0, "Ndn", s1;
    }

    auto Ntot = MPOT(nmpo);

    //
    // density matrix for test ****************
    //  
    int corrleft = 2;
    int corrright = 5;
    int lind = 2*corrleft-1;
    int rind = 2*corrright-1;
    auto dmpoup = AutoMPO(sites);
    auto dmpodn = AutoMPO(sites);
    auto dompo  = AutoMPO(sites);
    dmpoup += 1.0, "Cdagup", lind, "Cup", rind;
    dmpodn += 1.0, "Cdagdn", lind, "Cdn", rind;
    dompo  += 1.0, "Nupdn", lind;
    auto Dup = MPOT(dmpoup);
    auto Ddn = MPOT(dmpodn);
    auto Do  = MPOT(dompo);
        


    // END TEST*********************************
    
    ////////////////////////////////////////////////////
    //                 time evolution                 //
    ////////////////////////////////////////////////////

    MPOT expHa,expHb;
    MPOT expH;

    if(realstep)
        {
        expH = toExpH<TensorT>(ampo,tau);
        }
    else
        {
        auto taua = tau/2.*(1.+1._i);
        auto taub = tau/2.*(1.-1._i);
        println("Making expHa and expHb");
        expHa = toExpH<TensorT>(ampo,taua);
        expHb = toExpH<TensorT>(ampo,taub);
        }



    //
    // Make initial 'wavefunction' which is a product
    // of perfect singlets between neighboring sites
    //
    auto psi = MPST(sites);

    for(int n = 1; n <= 2*N; n += 2)
        {
        auto s1 = sites(n);
        auto s2 = sites(n+1);
        auto wf = TensorT(s1,s2);
        // define the initial state of the real-facticious pair for fermions
        //wf.set(s1(1),s2(4), 0.5);
        //wf.set(s1(2),s2(3), 0.5);
        //wf.set(s1(3),s2(2), 0.5);
        //wf.set(s1(4),s2(1), 0.5);
        wf.set(s1(1),s2(1), 0.5);
        wf.set(s1(2),s2(2), 0.5);
        wf.set(s1(3),s2(3), 0.5);
        wf.set(s1(4),s2(4), 0.5);
        TensorT D;
        psi.Aref(n) = TensorT(s1);
        psi.Aref(n+1) = TensorT(s2);
        svd(wf,psi.Aref(n),D,psi.Aref(n+1));
        psi.Aref(n) *= D;
        }

    auto obs = TStateObserver<TensorT>(psi);

    auto ttotal = beta/2.;
    const int nt = int(ttotal/tau+(1e-9*(ttotal/tau)));
    if(fabs(nt*tau-ttotal) > 1E-9)
        {
        Error("Timestep not commensurate with total time");
        }
    printfln("Doing %d steps of tau=%f",nt,tau);

    auto targs = args;

    auto En = Vector(nt);
    auto Nn = Vector(nt);
    auto Betas = Vector(nt);

    Real tsofar = 0;
    for(int tt = 1; tt <= nt; ++tt)
        {
        if(realstep)
            {
            psi = exactApplyMPO(expH,psi,args);
            }
        else
            {
            psi = exactApplyMPO(expHa,psi,args);
            psi = exactApplyMPO(expHb,psi,args);
            }
        psi.Aref(1) /= norm(psi.A(1));
        //cout << "norm of psi: " << norm(psi.Aref(1)) << endl;
        tsofar += tau;
        targs.add("TimeStepNum",tt);
        targs.add("Time",tsofar);
        targs.add("TotalTime",ttotal);
        obs.measure(targs);

        //Record beta value
        auto bb = (2*tsofar);
        Betas(tt-1) = bb;

        //
        // Measure Energy and
        //
        auto en = overlap(psi,H,psi);
        printfln("\nEnergy/N %.4f %.20f",bb,en/N);
        En(tt-1) = en/N;

        //
        // Measure total particle number
        //

        auto npart = overlap(psi, Ntot, psi);
        printfln("\nNtot %.4f  %.6f", bb, npart);
        Nn(tt-1) = npart;

        /////////////////////////////////////// TEST ////////////////////////////////////////

        auto** rdm1up = get_rdm1up(psi, N);
        auto** rdm1dn = get_rdm1dn(psi, N);
        auto*** rdm1s = get_rdm1s(psi, N);
        cout << "spin up:    " << rdm1up[corrleft-1][corrright-1] << endl;
        cout << "spin up(new):    " << rdm1s[0][corrleft-1][corrright-1] << endl;
        cout << "spin down:  " << rdm1dn[corrleft-1][corrright-1] << endl;
        cout << "spin down(new):  " << rdm1s[1][corrleft-1][corrright-1] << endl;

        //
        // Measure density matrix up
        //
        auto d12 = overlap(psi, Dup, psi);
        printfln("\n <a^+_1 a_2> UP  %.4f %.20f",bb,d12);
        println();

        // Measure density matrix down
        //
        auto ddn = overlap(psi, Ddn, psi);
        printfln("\n <a^+_1 a_2> DOWN  %.4f %.20f",bb,ddn);
        println();


        auto dos = overlap(psi, Do, psi);
        printfln("\n Do from mpo %.4f %.20f",bb, dos);
        auto* rdm2=get_rdm2diag(psi, N);
        cout << "Double occ: " << rdm2[corrleft-1] << endl;
        
        ////////////////////////////////////END TEST ////////////////////////////////////////
        
        }

    std::ofstream enf("/home/sunchong/work/finiteTMPS/tests/chkdr/en_U" + std::to_string(U) + ".dat");
    std::ofstream npartf("/home/sunchong/work/finiteTMPS/tests/chkdr/npart_U" + std::to_string(U) + ".dat");
    //std::ofstream susf("sus.dat");
    for(auto n : range(Betas))
        {
        enf << format("%.14f %.14f\n",Betas(n),En(n));
        npartf << format("%.14f %.14f\n",Betas(n),Nn(n));
        //susf << format("%.14f %.14f\n",Betas(n),Sus(n));
        }
    enf.close();
    npartf.close();
    //susf.close();

    writeToFile("chkdr/sites",sites);
    writeToFile("chkdr/psi",psi);

    return 0;
    }

//////////////////////////////////////////////////////////////////////
//              1-particle reduced density matrix (1RDM)            //
//////////////////////////////////////////////////////////////////////

//
//Spin up
//
double** get_rdm1up(MPST psi, int N)
    {
    // initialize
    auto sites = psi.sites();
    double** rdm1 = 0;
    rdm1 = new double*[N];
    for(int i=0; i<N; ++i)
        {
        rdm1[i] = new double[N];
        }
    
    // off-diagonal terms
    int lind, rind, k;
    for(int i=1; i<N; ++i)
        {
        lind = 2*i-1;
        auto AdagupF_i = sites.op("Adagup*F", lind);
        psi.position(lind);
        auto ir = commonIndex(psi.A(lind), psi.A(lind+1), Link);
        auto Corrup = psi.A(lind)*AdagupF_i*dag(prime(psi.A(lind),Site,ir));
        for(int j=i+1; j<=N; ++j)
            {
            rind = 2*j-1;
            auto Aup_j = sites.op("Aup", rind);
            //first apply F to the ancilla - (rind-1) site
            k = 2*j-2;
            Corrup *= psi.A(k);
            Corrup *= sites.op("F", k);
            Corrup *= dag(prime(psi.A(k)));
            //measure the correlation function
            auto Corrij = Corrup * psi.A(rind);
            Corrij *= Aup_j;
            auto jl = commonIndex(psi.A(rind), psi.A(rind-1), Link);
            Corrij *= dag(prime(psi.A(rind),jl,Site));
            auto dij = Corrij.real();
            rdm1[i-1][j-1] = dij;
            rdm1[j-1][i-1] = dij;
            // apply F to the rind site
            k = rind;
            Corrup *= psi.A(k);
            Corrup *= sites.op("F", k);
            Corrup *= dag(prime(psi.A(k)));
            }
        }
    
    // diagonal terms
    for (int i=1; i<=N; ++i)
        {
        int ind = 2*i-1;
        psi.position(ind);
        auto res = psi.A(ind)*sites.op("Nup", ind)*dag(prime(psi.A(ind), Site));
        rdm1[i-1][i-1] = res.real();
        }
    
    return rdm1;

    }

//
//Spin down
//
double** get_rdm1dn(MPST psi, int N)
    {
    // initialize
    auto sites = psi.sites();
    double** rdm1 = 0;
    rdm1 = new double*[N];
    for(int i=0; i<N; ++i)
        {
        rdm1[i] = new double[N];
        }
    
    // off-diagonal terms
    int lind, rind, k;
    for(int i=1; i<N; ++i)
        {
        lind = 2*i-1;
        auto Adagdn_i = sites.op("Adagdn", lind);
        psi.position(lind);
        auto ir = commonIndex(psi.A(lind), psi.A(lind+1), Link);
        auto Corrdn = psi.A(lind)*Adagdn_i*dag(prime(psi.A(lind),Site,ir));
        for(int j=i+1; j<=N; ++j)
            {
            rind = 2*j-1;
            auto AdnF_j = sites.op("F*Adn", rind);
            //first apply F to the ancilla - (rind-1) site
            k = 2*j-2;
            Corrdn *= psi.A(k);
            Corrdn *= sites.op("F", k);
            Corrdn *= dag(prime(psi.A(k)));
            //measure the correlation function
            auto Corrij = Corrdn * psi.A(rind);
            Corrij *= AdnF_j;
            auto jl = commonIndex(psi.A(rind), psi.A(rind-1), Link);
            Corrij *= dag(prime(psi.A(rind),jl,Site));
            auto dij = Corrij.real();
            rdm1[i-1][j-1] = dij;
            rdm1[j-1][i-1] = dij;
            // apply F to the rind site
            k = rind;
            Corrdn *= psi.A(k);
            Corrdn *= sites.op("F", k);
            Corrdn *= dag(prime(psi.A(k)));
            }
        }
    
    // diagonal terms
    for (int i=1; i<=N; ++i)
        {
        int ind = 2*i-1;
        psi.position(ind);
        auto res = psi.A(ind)*sites.op("Ndn", ind)*dag(prime(psi.A(ind), Site));
        rdm1[i-1][i-1] = res.real();
        }
    
    return rdm1;
    }


//////////////////////////////////////////////////////////////////////
//         2-particle reduced density matrix (2RDM-diagonal)        //
//////////////////////////////////////////////////////////////////////

double *get_rdm2diag(MPST psi, int N)
    {
    auto sites = psi.sites();
    double* rdm2 = 0;
    rdm2 = new double[N];
    for(int i=1; i<=N; ++i)
        {
        int ind = 2*i-1;
        psi.position(ind);
        auto res = psi.A(ind)*sites.op("Nupdn", ind)*dag(prime(psi.A(ind), Site));
        rdm2[i-1] = res.real();
        }

    return rdm2;
    }


double*** get_rdm1s(MPST psi, int N)
    {
    // N is the number of sites
    // return a (2,N,N) array with to store the 1RDM. 2 accounts for spin freedom
    // initialization
    auto sites = psi.sites();
    double*** rdm1 = 0;
    rdm1 = new double**[2];
    for(int i=0; i<2; ++i)
        {
        rdm1[i] = new double*[N];
        for(int j=0; j<N; ++j)
            rdm1[i][j] = new double[N];
        }
    
    // off-diagonal terms
    int lind, rind, k;
    for(int i=1; i<N; ++i)
        {
        lind = 2*i-1;
        auto AdagupF_i = sites.op("Adagup*F", lind);
        auto Adagdn_i = sites.op("Adagdn", lind);
        psi.position(lind);
        auto ir = commonIndex(psi.A(lind), psi.A(lind+1), Link);
        auto Corrup = psi.A(lind)*AdagupF_i*dag(prime(psi.A(lind),Site,ir));
        auto Corrdn = psi.A(lind)*Adagdn_i*dag(prime(psi.A(lind),Site,ir));
        for(int j=i+1; j<=N; ++j)
            {
            rind = 2*j-1;
            auto Aup_j = sites.op("Aup", rind);
            auto AdnF_j = sites.op("F*Adn", rind);
            //first apply F to the ancilla - (rind-1) site
            k = 2*j-2;
            Corrup *= psi.A(k);
            Corrup *= sites.op("F", k);
            Corrup *= dag(prime(psi.A(k)));
            Corrdn *= psi.A(k);
            Corrdn *= sites.op("F", k);
            Corrdn *= dag(prime(psi.A(k)));
            //measure the correlation function
            auto Corrupij = Corrup * psi.A(rind);
            Corrupij *= Aup_j;
            auto Corrdnij = Corrdn * psi.A(rind);
            Corrdnij *= AdnF_j;
            auto jl = commonIndex(psi.A(rind), psi.A(rind-1), Link);
            Corrupij *= dag(prime(psi.A(rind),jl,Site));
            Corrdnij *= dag(prime(psi.A(rind),jl,Site));
            
            rdm1[0][i-1][j-1] = Corrupij.real();
            rdm1[0][j-1][i-1] = Corrupij.real();
            rdm1[1][i-1][j-1] = Corrdnij.real();
            rdm1[1][j-1][i-1] = Corrdnij.real();
            // apply F to the rind site
            k = rind;
            Corrup *= psi.A(k);
            Corrup *= sites.op("F", k);
            Corrup *= dag(prime(psi.A(k)));
            Corrdn *= psi.A(k);
            Corrdn *= sites.op("F", k);
            Corrdn *= dag(prime(psi.A(k)));
            }
        }
    
    // diagonal terms
    for (int i=1; i<=N; ++i)
        {
        int ind = 2*i-1;
        psi.position(ind);

        auto resup = psi.A(ind)*sites.op("Nup", ind)*dag(prime(psi.A(ind), Site));
        auto resdn = psi.A(ind)*sites.op("Ndn", ind)*dag(prime(psi.A(ind), Site));
        rdm1[0][i-1][i-1] = resup.real();
        rdm1[1][i-1][i-1] = resdn.real();
        }
    
    return rdm1;

    }



#include "itensor/all.h"
#include <iostream>
#include "TStateObserver.h"
//#include "S2.h"

using namespace std;
using namespace itensor;

using TensorT = ITensor;
using MPOT = MPOt<TensorT>;
using MPST = MPSt<TensorT>;

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

    auto Nx = input.getInt("Nx",10);
    auto Ny = input.getInt("Ny",1);
    auto Npart = input.getInt("Npart",1);
    auto U = input.getReal("U", 1.0);
    //auto V1 = input.getReal("V1",1.0);
    auto t1 = input.getReal("t1", 1.0);
    //auto t2 = input.getReal("t2", 1.0);
    auto periodic = input.getYesNo("periodic",false);
    auto lattice_type = input.getString("lattice_type","square");

    auto beta = input.getReal("beta",1);
    auto tau = input.getReal("tau",0.005);

    auto maxm = input.getInt("maxm",1000);
    auto cutoff = input.getReal("cutoff",1E-11);

//    auto Jz = input.getReal("Jz",1.);
//    auto Jxy = input.getReal("Jxy",1.);

    auto realstep = input.getYesNo("realstep",false);
    auto verbose = input.getYesNo("verbose",false);

    auto N = Nx*Ny;
    cout << "tau = " << tau << endl;

    Args args;
    args.add("Nx",Nx);
    args.add("Ny",Ny);
    //args.add("Npart",Npart);
    //args.add("Jz",Jz);
    //args.add("Jxy",Jxy);
    args.add("Maxm",maxm);
    args.add("Cutoff",cutoff);
    args.add("YPeriodic",periodic);
    args.add("Verbose",verbose);

    auto sites = Hubbard(2*N, {"ConserveNf",false,"ConserveSz", true});

    //auto sites = Hubbard(2*N);
    //auto Ntot_3 = sites.op("Ntot", N);
    writeToFile("chkdr/sites",sites);

    LatticeGraph lattice; 
    if(lattice_type == "triangular")
        lattice = triangularLattice(Nx,Ny,args);
    else if(lattice_type == "square")
        lattice = squareLattice(Nx,Ny,args);

    auto ampo = AutoMPO(sites);

    //////////////////
    // Coulomb term //
    //////////////////
    for(int i=1; i<=N;++i ) 
    {
        int s1 = 2*i-1;
    //    //cout << s1 << endl;
        ampo += U,"Nup",s1, "Ndn", s1;
    }

    ///////////////////////////
    // nearest neighbor term //
    ///////////////////////////
    for(auto b : lattice)
        {
        
        //cout << b.s1 << " " << b.s2 << endl;

        auto s1 = 2*b.s1 - 1;
        auto s2 = 2*b.s2 - 1;

        //cout << s1 << " " << s2 << endl; 
        ampo += -t1,"Adagup",s1,"Aup",s2;
        ampo += -t1,"Adagup",s2,"Aup",s1;
        ampo += -t1,"Adagdn",s1,"Adn",s2;
        ampo += -t1,"Adagdn",s2,"Adn",s1;
        //ampo += V1,"Ntot",b,"Ntot",b+1;
        //ampo += U, "Nup", s1, "Ndn", s1;
        }
    //cout << ampo << endl;

    ///////////////////////////////////
    // next-to-nearest neighbor term //
    ///////////////////////////////////
    //for(int b = 1; b < N-1; ++b)
    //    {
    //    ampo += -t2,"Cdagup",b,"Cup",b+2;
    //    ampo += -t2,"Cdagup",b+2,"Cup",b;
    //    ampo += -t2,"Cdagdn",b,"Cdn",b+2;
    //    ampo += -t2,"Cdagdn",b+2,"Cdn",b;
    //    }

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

    auto H = MPOT(ampo);


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
    //auto Sus = Vector(nt);
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
        tsofar += tau;
        targs.add("TimeStepNum",tt);
        targs.add("Time",tsofar);
        targs.add("TotalTime",ttotal);
        obs.measure(targs);

        //Record beta value
        auto bb = (2*tsofar);
        Betas(tt-1) = bb;

        //
        // Measure Energy
        //
        auto en = overlap(psi,H,psi);
        printfln("\nEnergy/N %.4f %.20f",bb,en/(2*N));
        En(tt-1) = en/N;

        //
        // Measure Susceptibility
        //
        //auto s2val = overlap(psi,S2,psi);
        //Sus(tt-1) = (s2val*bb/3.)/N;

        println();
        }

    std::ofstream enf("en.dat");
    //std::ofstream susf("sus.dat");
    for(auto n : range(Betas))
        {
        enf << format("%.14f %.14f\n",Betas(n),En(n));
        //susf << format("%.14f %.14f\n",Betas(n),Sus(n));
        }
    enf.close();
    //susf.close();

    writeToFile("sites",sites);
    writeToFile("psi",psi);

    return 0;
    }


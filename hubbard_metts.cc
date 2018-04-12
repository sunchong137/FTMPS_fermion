#include "itensor/all.h"
#include "basis/rotatexz.h"
#include "heisops.h"
#include "limits.h"
#include "collapse.h"
#include "S2.h"
#include "trotter.h"
#include "TStateObserver.h"

using namespace std;
using namespace itensor;

using TensorT = ITensor;
using MPOT = MPOt<TensorT>;
using MPST = MPSt<TensorT>;

int 
main(int argc, char* argv[])
    {
    if(argc < 2) 
        { 
        printfln("Usage: %s <input_file>", argv[0]); 
        return 0; 
        }
    println("Process id is ",getpid());

    auto infile = InputFile(argv[1]);
    println(infile,"******************\n");
    auto in = InputGroup(infile,"input");

    auto Nx = in.getInt("Nx");
    auto Ny = in.getInt("Ny");
    auto U = in.getReal("U", 1.0);
    auto t1 = in.getReal("t1", 1.0);
    auto beta = in.getReal("beta");
    auto cutoff = in.getReal("cutoff");
    auto realstep = in.getYesNo("realstep",false);
    

    auto hz = in.getReal("hz",0);
    auto nmetts = in.getInt("nmetts",50000);
    auto maxm = in.getInt("maxm",5000);
    auto tau = in.getReal("tau",0.1);
    auto nwarm = in.getInt("nwarm",5);
    
    //Real Jxy = 1;
    //Real Jz = 1;

    string lattice_type = "square";

    auto N = Nx*Ny;
    auto Npart = N;

    auto sites = Hubbard(N);
        
    auto basis = rotateXZ<IQTensor>(sites);
        
    Args args;
    args.add("Nx",Nx);
    args.add("Ny",Ny);
    args.add("hz",hz);
    args.add("Maxm",maxm);
    args.add("Cutoff",cutoff);
    args.add("YPeriodic",false);
    args.add("hx",0.);

    Print(args);

    // Lattice bonds
    auto lattice = squareLattice(Nx,Ny,args);

    // Hamiltonian via AutoMPO
    auto ampo = AutoMPO(sites);

    for(int i=1; i<=N;++i ) 
    {
        ampo += U, "Nupdn", i;
    }

    for(auto b : lattice)
        {
        auto s1 = b.s1,
        s2 = b.s2;
        ampo += -t1,"Cdagup",s1,"Cup",s2;
        ampo += -t1,"Cdagup",s2,"Cup",s1;
        ampo += -t1,"Cdagdn",s1,"Cdn",s2;
        ampo += -t1,"Cdagdn",s2,"Cdn",s1;
        }
    
    auto mu = U/2;
    for(int i=1; i<=N; ++i) 
    {
        ampo += -mu, "Nup", i;
        ampo += -mu, "Ndn", i;
    }
        
    //auto H = MPOT(ampo);
    auto H = IQMPO(ampo);

    IQMPO expHa,expHb;
    IQMPO expH;

    if(realstep)
        {
        //expH = toExpH<TensorT>(ampo,tau);
        expH = toExpH<IQTensor>(ampo,tau);
        }
    else
        {
        auto taua = tau/2.*(1.+1._i);
        auto taub = tau/2.*(1.-1._i);
        println("Making expHa and expHb");
        expHa = toExpH<IQTensor>(ampo,taua);
        expHb = toExpH<IQTensor>(ampo,taub);
        }



    
    //IQMPO H2;
    //nmultMPO(H,H,H2,"Cutoff=1E-12,Maxm=200");
        
    //IQMPO S2 = makeS2(sites);
    //IQMPO Sxy2 = makeSxy2(sites);
    //IQMPO Sz2 = makeTotSz2(sites);
  
    //auto ops = HeisOps(sites,Nx,Ny,args);
    //auto gates = makeGates<IQTensor>(sites,lattice,tau,ops);


    auto state = InitState(sites,"Up");
    for (int i = 1; i <= Nx; ++i)
    for (int j = 1; j <= Ny; ++j)
        {
        auto st = ((i+j)%2==0 ? "Up" : "Dn");
        int x=(i-1)*Ny+j;
        state.set(x,st);
        }
    println();

    auto psi = IQMPS(state);
    //auto psi = MPST(state);

    //observables
    bool verbose = true;
    Stats en_stat,
    //      magz_stat,
    //      magz2_stat,
    //      en2_stat,
    //      s2_stat,
    //      sxy2_stat,
    //      z2_stat,
          cpu_stat;
        
    Args targs;
    targs.add("Verbose",false);
    targs.add("Maxm",maxm);
    targs.add("Minm",6);
    targs.add("Cutoff",cutoff);
        
    auto obs = TStateObserver<IQTensor>(psi);
    //auto obs = TStateObserver<TensorT>(psi);

    for(int step = 1; step <= (nwarm+nmetts); ++step)
        {
        psi.position(1);
        args.add("Step",step);

        if(verbose)
            {
            if (step <= nwarm) 
                printfln("\nStarting step %d (warmup %d/%d)",step,step,nwarm);
            else
                printfln("\nMaking METTS number %d/%d",step-nwarm,nmetts);
            }

        println("Doing regular imaginary time evolution");
        auto cpu_time_1s = cpu_mytime();
        //------Replacing gateTEvol------
        //gateTEvol(gates,beta/2.,tau,psi,obs,targs);
        auto ttotal = beta/2.;
        const int nt = int(ttotal/tau+(1e-9*(ttotal/tau)));
        for(int tt=1; tt<=nt; ++tt)
            {
            
            printfln("time step %d", tt);
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
            }
        //-------------------------------
        auto cpu_time_1e = cpu_mytime();
        printfln("CPU time for generation of METTS %.14f",cpu_time_1e - cpu_time_1s);


        if(step > nwarm) println("\nDone making METTS ",step-nwarm);
        int maxmm = 0;
        for(int b = 0; b < psi.N(); ++b)
            {
            int m_b = linkInd(psi,b).m();
            maxmm = std::max(maxmm,m_b);
            }
            
        if(step > nwarm)
            {
            //
            //CPU time
            //
            cpu_stat.putin(cpu_time_1e-cpu_time_1s);
            printfln("Average CPU time = %.14f %.3E",cpu_stat.avg(),cpu_stat.err());
            
            //
            //Energy
            //
            const auto en = psiHphi(psi,H,psi);
            //auto en = overlap(psi,H,psi);
            en_stat.putin(en);
            auto avgEn = en_stat.avg();
            printfln("Energy of METTS %d = %.14f",step-nwarm,en);
            printfln("Average energy = %.14f %.3E",avgEn,en_stat.err());
            printfln("Average energy per site = %.14f %.3E",avgEn/N,en_stat.err()/N);
            
            //
            //Specific heat
            //
            //const auto en2 = psiHphi(psi,H2,psi);
            //en2_stat.putin(en2);
            //auto avgEn2 = en2_stat.avg();
            //printfln("<H^2> for METTS %d = %.14f",step-nwarm,en2);
            //printfln("Average specific heat = %.14f %.3E",(avgEn2-sqr(avgEn))*sqr(beta),en2_stat.err());
            //printfln("Average specific heat per site = %.14f %.3E",(avgEn2-sqr(avgEn))*sqr(beta)/N,en2_stat.err()/N);
            //
            ////
            ////Susceptibility
            ////
            //const auto s2val = overlap(psi,S2,psi);
            //s2_stat.putin(s2val);
            //auto asus = (s2_stat.avg()*beta/3);
            //auto esus = (s2_stat.err()*beta/3);
            //printfln("<S^2> for METTS %d = %.14f",step-nwarm,s2val);
            //printfln("Average total susceptibility = %.14f %.3E",asus,esus);
            //printfln("Average total susceptibility per site = %.14f %.3E (%.5f,%.5f)",
            //         asus/N,esus/N,(asus-esus)/N,(asus+esus)/N);

            //const auto sxy2val = overlap(psi,Sxy2,psi);
            //sxy2_stat.putin(sxy2val);
            //asus = (sxy2_stat.avg()*beta/2);
            //esus = (sxy2_stat.err()*beta/2);
            //printfln("<(Sx^2+Sy^2)> for METTS %d = %.14f",step-nwarm,sxy2val);
            //printfln("Average total XY susceptibility = %.14f %.3E",asus,esus);
            //printfln("Average total XY susceptibility per site = %.14f %.3E (%.5f,%.5f)",
            //         asus/N,esus/N,(asus-esus)/N,(asus+esus)/N);


            }

        // Collapse into product state
        auto cps = collapse(psi,basis,args);
        for(int j = 1; j <= N; ++j)
            {
            print(basis->statestr(j,cps[j],args)," ");
            }
        println();
        }

    return 0;
    }


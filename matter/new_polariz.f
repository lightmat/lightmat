      Program Polarizability_New_Data
      implicit none
      real*8 lam 
      character*8 tstart,tend
      external Magic,Data6S,Data6P
      common /SH/ lam
      call time(tstart)
      write(*,*) "Started at ",tstart
	
C     DATA
      call Data6S
      call Data6P

C     Magic wavelength
      call Magic

      call time(tend)
      write(*,*) "Ended at ",tend
      stop
      End

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Subroutine Data6S
      implicit none
      real*8 d1(6:40),d2(6:40),E1(6:40),E2(6:40),gam1(6:40),gam2(6:40)
      common /S/ d1,d2,E1,E2,gam1,gam2

C     6S1/2 - nP1/2 

      d1(6)= -4.489D0     
      d1(7)= -0.276D0      
      d1(8)= 8.1D-2
      d1(9)= 4.3D-2
      d1(10)=-4.7D-2
      d1(11)=-3.4D-2          
      d1(12)=-2.6D-2
      d1(13)= 2.1D-2      
      d1(14)=-1.7D-2
      d1(15)= 1.5D-2     
      d1(16)= 2.2D-2
      d1(17)=-2.3D-2
      d1(18)= 1.9D-2
      d1(19)=-1.0D-2
      d1(20)=-3.5D-2
      d1(21)=-2.0D-3
      d1(22)=-2.7D-2
      d1(23)=-2.0D-3
      d1(24)= 0.0D0
      d1(25)= 3.1D-2
      d1(26)=-4.5D-2
      d1(27)=-4.4D-2
      d1(28)=-4.0D-2
      d1(29)=-3.5D-2
      d1(30)=-2.9D-2
      d1(31)=-2.2D-2
      d1(32)= 1.5D-2
      d1(33)=-1.0D-2
      d1(34)= 7.0D-3
      d1(35)=-4.0D-3
      d1(36)= 2.0D-3
      d1(37)= 0.0D0
      d1(38)=-1.0D-3
      d1(39)= 1.0D-3
      d1(40)= 0.0D0
     
 !    Experimental energy values
      E1(6)= 5.093193103836056D-002
      E1(7)= 9.917021023503879D-002
      E1(8)= 0.117138059058354     
      E1(9)= 0.125923406466419     
      E1(10)= 0.130888971551178     
      E1(11)= 0.133971836814040     
      E1(12)= 0.136017669501871     
      E1(13)= 0.137444880882430     
      E1(14)= 0.138480096391185     
      E1(15)= 0.139254842329305     
      E1(16)= 0.139849741216678     
      E1(17)= 0.140316459610575     
      E1(18)= 0.140689352730228     
      E1(19)= 0.140991988945859     
      E1(20)= 0.141240862371317     
      E1(21)= 0.141448281079075     
      E1(22)= 0.141622719213779     
      E1(23)= 0.141770888620940     
      E1(24)= 0.141897810199139     
      E1(25)= 0.142007362118216           
!     End of experimental values      
                 
      E1(26)=0.363380732288118     
      E1(27)=0.564921433435668     
      E1(28)=0.915961845916200     
      E1(29)= 1.52296420894661     
      E1(30)= 2.56822599416345     
      E1(31)= 4.36544284165791     
      E1(32)= 7.45558489203416     
      E1(33)= 12.7798159463226     
      E1(34)= 21.9954136484641     
      E1(35)= 37.9759717082086     
      E1(36)= 65.5424424671927     
      E1(37)= 86.8849129360100     
      E1(38)= 112.705059998209     
      E1(39)= 192.564589339110     
      E1(40)= 325.701404691948 
      
      gam1=0.d0
      gam1(6)= 6.966386007120775D-010
      gam1(7)= 1.519059170997169D-010
      gam1(8)= 6.651931083188239D-011
      gam1(9)= 3.700892566282911D-011
      gam1(10)= 2.302777596798256D-011
      gam1(11)= 1.538410243239171D-011
         
C     6S1/2 - nP3/2
                 
      d2(6)= -6.324D0      
      d2(7)= -0.586D0      
      d2(8)= 0.218D0      
      d2(9)= 0.127D0      
      d2(10)=0.114D0      
      d2(11)=-8.5D-2           
      d2(12)=-6.7D-2      
      d2(13)=5.5D-2
      d2(14)=4.6D-2
      d2(15)=-3.9D-2    
      d2(16)=-6.2D-2
      d2(17)=6.5D-2
      d2(18)=6.2D-2
      d2(19)=3.9D-2
      d2(20)=-0.1120D0     
      d2(21)=-1.4D-2
      d2(22)=-0.1190D0     
      d2(23)=8.1D-2
      d2(24)=4.0D-3
      d2(25)=3.2D-2
      d2(26)=1.0D-3
      d2(27)=-1.7D-2
      d2(28)=-2.5D-2
      d2(29)=2.9D-2
      d2(30)=2.9D-2
      d2(31)=-2.3D-2
      d2(32)=-1.7D-2
      d2(33)=-1.2D-2
      d2(34)=-8.0D-3
      d2(35)=5.0D-3
      d2(36)=-3.0D-3
      d2(37)=0.0D0
      d2(38)=2.0D-3
      d2(39)=-1.0D-3
      d2(40)=-1.0D-3
                       
 !    Experimental energy values
      E2(6)= 5.345631790751226D-002
      E2(7)= 9.999513007518303D-002
      E2(8)= 0.117514742729530     
      E2(9)= 0.126126990790714     
      E2(10)= 0.131011375660218     
      E2(11)= 0.134051121138430     
      E2(12)= 0.136071951258566     
      E2(13)= 0.137483654106035     
      E2(14)= 0.138508761067326     
      E2(15)= 0.139276617371754     
      E2(16)= 0.139866676383724     
      E2(17)= 0.140329877925124     
      E2(18)= 0.140700174890832     
      E2(19)= 0.141000848328603     
      E2(20)= 0.141248198070175     
      E2(21)= 0.141454436003797     
      E2(22)= 0.141628059693674     
      E2(23)= 0.141775322253541     
      E2(24)= 0.141901758718786     
      E2(25)= 0.142010666827770     
!     End of experimental values      
               
      E2(26)=0.376973139285157     
      E2(27)=0.588894302218703     
      E2(28)=0.957990590680449     
      E2(29)= 1.59617893405377     
      E2(30)= 2.69538903157404     
      E2(31)= 4.58648830455661     
      E2(32)= 7.84141386034317     
      E2(33)= 13.4576560688759     
      E2(34)= 23.1901747728089     
      E2(35)= 40.0714529286711     
      E2(36)= 69.1923751994882     
      E2(37)= 86.8853139531241     
      E2(38)= 119.012977696105     
      E2(39)= 203.329003839367     
      E2(40)= 343.797587842136 
            
      gam2=0.d0      
      gam2(6)= 7.977479531765387D-010
      gam2(7)= 1.806906370596951D-010
      gam2(8)= 7.595295854985846D-011
      gam2(9)= 4.063725170820452D-011
      gam2(10)= 2.467261710855274D-011
      gam2(11)= 1.630327836388681D-011

                
	return  
	End

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Subroutine Data6P
      implicit none
      real*8 d1(5:42),d2(5:42),d3(5:42),E1(5:42),E2(5:42),E3(5:42),
     & gam1(5:42),gam2(5:42),gam3(5:42)
      common /P/ d1,d2,d3,E1,E2,E3,gam1,gam2,gam3

C     6P3/2 - nS1/2 
      
      d1= 0.d0
      d1(6)=-6.324D0     
      d1(7)=-6.470D0     
      d1(8)=-1.461D0     
      d1(9)=0.770D0     
      d1(10)=-0.509D0     
      d1(11)= 0.381D0     
      d1(12)=-0.297D0     
      d1(13)=-0.241D0     
      d1(14)=-0.219D0     
      d1(15)= 0.234D0     
      d1(16)= 0.251D0     
      d1(17)=-0.259D0     
      d1(18)= 0.239D0     
      d1(19)= 0.376D0     
      d1(20)=-0.213D0     
      d1(21)= 0.349D0     
      d1(22)= 0.472D0     
      d1(23)= 0.578D0     
      d1(24)=-2.6D-2
      d1(25)=-0.464D0     
      d1(26)= 0.325D0     
      d1(27)= 0.211D0     
      d1(28)=-0.132D0     
      d1(29)= 8.1D-2
      d1(30)= 4.8D-2
      d1(31)=-2.7D-2
      d1(32)= 1.5D-2
      d1(33)=-8.0D-3
      d1(34)= 4.0D-3
      d1(35)= 2.0D-3
      d1(36)=-1.0D-3
      d1(37)= 0.0D0
      d1(38)=-1.0D-3
      d1(39)= 0.0D0
      d1(40)= 0.0D0

      E1= 1.d20
                
 !    Experimental energy values
      E1(6)= -5.345631790751226D-002
      E1(7)= 3.099775413762914D-002
      E1(8)= 5.734075358037974D-002
      E1(9)= 6.915766821038563D-002
      E1(10)= 7.548899598678790D-002
      E1(11)= 7.927759439093549D-002
      E1(12)= 8.172468019245956D-002
      E1(13)= 8.339678668040110D-002
      E1(14)= 8.458989362990918D-002
      E1(15)= 8.547099016078509D-002
      E1(16)= 8.614013028461728D-002
      E1(17)= 8.666024208663467D-002
      E1(18)= 8.707251578930782D-002
      E1(19)= 8.740483146794366D-002
      E1(20)= 8.767660644404056D-002
      E1(21)= 8.790170400379406D-002
      E1(22)= 8.809023151002081D-002
      E1(23)= 8.824970304208259D-002
      E1(24)= 8.838579779779630D-002
      E1(25)= 8.850287063073646D-002      
!     End of experimental values      
          
      E1(26)= 0.284925439315096     
      E1(27)= 0.466119891933841     
      E1(28)= 0.787217482469064     
      E1(29)=  1.35215401347409     
      E1(30)=  2.34166762828119     
      E1(31)=  4.07068483727570     
      E1(32)=  7.09085178172705     
      E1(33)=  12.3801063564679     
      E1(34)=  21.6970346350570     
      E1(35)=  38.1633122249160     
      E1(36)=  67.1643152115890     
      E1(37)=  86.8415742191157     
      E1(38)=  117.918483664165     
      E1(39)=  205.937860968824     
      E1(40)=  356.237985030300     

      gam1= 0.d0
      gam1(6)= 7.977505628881344D-010
      gam1(7)= 1.279108484907938D-009
      gam1(8)= 1.039638965913161D-009
      gam1(9)= 9.450606003303758D-010
      gam1(10)= 8.860398299922694D-010
      gam1(11)= 8.543524491959909D-010
      gam1(12)= 8.362108189691137D-010
      gam1(13)= 8.248420640269375D-010
      gam1(14)= 8.176096007764892D-010
      gam1(15)= 8.126992661950811D-010

C     6P3/2 - nD3/2      

      d2(5)= 3.166D0     
      d2(6)= 2.100D0     
      d2(7)=0.976D0     
      d2(8)=0.607D0     
      d2(9)=0.391D0     
      d2(10)= 0.304D0    
      d2(11)= 0.246D0     
      d2(12)= 0.211D0     
      d2(13)= 0.215D0     
      d2(14)=-0.234D0     
      d2(15)=-0.248D0     
      d2(16)=-0.256D0     
      d2(17)= 0.269D0     
      d2(18)= 0.204D0     
      d2(19)= 0.397D0     
      d2(20)=-1.2D-2
      d2(21)=-0.482D0     
      d2(22)=-2.1D-2
      d2(23)= 0.491D0     
      d2(24)= 0.438D0     
      d2(25)= 0.344D0     
      d2(26)= 2.8D-2
      d2(27)=-0.244D0     
      d2(28)= 0.158D0     
      d2(29)= 9.5D-2
      d2(30)= 5.3D-2
      d2(31)= 2.7D-2
      d2(32)=-1.1D-2
      d2(33)=-2.0D-3
      d2(34)= 2.0D-3
      d2(35)=-4.0D-3
      d2(36)=-4.0D-3
      d2(37)=-4.0D-3
      d2(38)= 3.0D-3
      d2(39)=-2.0D-3
      d2(40)=-2.0D-3
      d2(41)=-1.0D-3
      d2(42)= 1.0D-3
           
 !    Experimental energy values
      E2(5)= 1.260714889797211D-002
      E2(6)= 4.946591092759109D-002
      E2(7)= 6.522633278024305D-002
      E2(8)= 7.326099980765223D-002
      E2(9)= 7.789680609211680D-002
      E2(10)= 8.081106418960049D-002
      E2(11)= 8.276141282397792D-002
      E2(12)= 8.413039274354839D-002
      E2(13)= 8.512803411930003D-002
      E2(14)= 8.587742997807847D-002
      E2(15)= 8.645459729666347D-002
      E2(16)= 8.690853580962894D-002
      E2(17)= 8.727198369302346D-002
      E2(18)= 8.756748569091641D-002
      E2(19)= 8.781097120503403D-002
      E2(20)= 8.801398143115806D-002
      E2(21)= 8.818501165532681D-002
      E2(22)= 8.833044370773041D-002
      E2(23)= 8.845514124783875D-002
      E2(24)= 8.856287760422926D-002
      E2(25)= 8.865656172322443D-002
!     End of experimental values      
                   
      E2(26)=0.191402624534708     
      E2(27)=0.254271269348370     
      E2(28)=0.353361342877603     
      E2(29)=0.503123540852616     
      E2(30)=0.728712496578961     
      E2(31)= 1.06760346240848     
      E2(32)= 1.57561773177378     
      E2(33)= 2.33604571866312     
      E2(34)= 3.47329285407881     
      E2(35)= 5.17294342551735     
      E2(36)= 7.71113485545201     
      E2(37)= 11.4983395685999     
      E2(38)= 17.1463099729236     
      E2(39)= 25.5705227322729     
      E2(40)= 38.1383857612701     
      E2(41)= 56.8696234492447     
      E2(42)= 84.7120778998163     

      gam2= 0.d0
      gam2(5)= 8.227860126012248D-010
      gam2(6)= 1.218636384151681D-009
      gam2(7)= 1.058990038155164D-009
      gam2(8)= 9.723939898722039D-010
      gam2(9)= 9.116800007129223D-010
      gam2(10)= 8.753967402591681D-010
      gam2(11)= 8.526592303748156D-010

C     6P3/2 - nD5/2      

      d3(5)= 9.590D0     
      d3(6)= 6.150D0     
      d3(7)= 2.890D0     
      d3(8)=-1.810D0     
      d3(9)= 1.169D0     
      d3(10)= 0.909D0     
      d3(11)=-0.735D0     
      d3(12)=-0.63D0     
      d3(13)=-0.642D0     
      d3(14)= 0.699D0     
      d3(15)= 0.741D0     
      d3(16)= 0.766D0     
      d3(17)= 0.798D0     
      d3(18)=-0.745D0     
      d3(19)=-0.903D0     
      d3(20)= 0.84D0     
      d3(21)= -1.438D0     
      d3(22)= 0.13D0     
      d3(23)= -1.456D0     
      d3(24)= -1.288D0     
      d3(25)=-0.149D0     
      d3(26)= 0.998D0     
      d3(27)= 0.713D0     
      d3(28)=-0.462D0     
      d3(29)= 0.28D0     
      d3(30)=-0.158D0     
      d3(31)= 8.1D-2
      d3(32)= 3.5D-2
      d3(33)=-8.D-3
      d3(34)=-5.D-3
      d3(35)=-1.D-2
      d3(36)=-1.1D-2
      d3(37)= 1.D-2
      d3(38)=-8.D-3
      d3(39)=-6.D-3
      d3(40)=-5.D-3
      d3(41)= 3.D-3
      d3(42)=-2.D-3
           
 !    Experimental energy values
      E3(5)= 1.305178118845034D-002
      E3(6)= 4.966121958118065D-002
      E3(7)= 6.532173696114883D-002
      E3(8)= 7.331403645476571D-002
      E3(9)= 7.792920126728947D-002
      E3(10)= 8.083226094205051D-002
      E3(11)= 8.277602298455662D-002
      E3(12)= 8.414088101594550D-002
      E3(13)= 8.513581383297451D-002
      E3(14)= 8.588335722275615D-002
      E3(15)= 8.645921505074987D-002
      E3(16)= 8.691220215546776D-002
      E3(17)= 8.727494180219729D-002
      E3(18)= 8.756990733724329D-002
      E3(19)= 8.781298651743284D-002
      E3(20)= 8.801567369942702D-002
      E3(21)= 8.818644585279864D-002
      E3(22)= 8.833166835936965D-002
      E3(23)= 8.845619408009664D-002
      E3(24)= 8.856377734364143D-002
      E3(25)= 8.865735880841595D-002
!     End of experimental values      
          
      E3(26)=0.194900624693274     
      E3(27)=0.262982635124555     
      E3(28)=0.366437357320160     
      E3(29)=0.522747764041982     
      E3(30)=0.758132172131350     
      E3(31)= 1.11166315339563     
      E3(32)= 1.64159706375991     
      E3(33)= 2.43495261231384     
      E3(34)= 3.62180721855539     
      E3(35)= 5.39634104132320     
      E3(36)= 8.04781852126053     
      E3(37)= 12.0069779761342     
      E3(38)= 17.9168119393320     
      E3(39)= 26.7397633303383     
      E3(40)= 39.9119863813775     
      E3(41)= 59.5546149025787     
      E3(42)= 88.7700174200515 

      gam3= 0.d0
      gam3(5)= 8.156744935522890D-010
      gam3(6)= 1.228311920272682D-009
      gam3(7)= 1.073503342336665D-009
      gam3(8)= 9.704588826480038D-010
      gam3(9)= 9.104705586977969D-010
      gam3(10)= 8.744291866470680D-010
      gam3(11)= 8.519335651657405D-010
           
	return  
	End

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Function alphaS0(w)
      implicit none
      integer q
      real*8 w,alphaS0,pi,c,hbar,eps0,ea,au,Eau,d1(6:40),d2(6:40),
     & E1(6:40),E2(6:40),gam1(6:40),gam2(6:40),E,SJ1,SJ2,JP,tmp
      common /S/ d1,d2,E1,E2,gam1,gam2
      pi=4*datan(1.D0)
      c=299792458.D0           ! velocity of light in m/s
      hbar=1.054571726D-34     ! reduced Planck constant in J s
      eps0=8.854187817D-12     ! permittivity of free space
      ea=1.602176565D-19*0.52917721092D-10 ! atomic unit  ea of dipole
	au=ea**2/4.35974434D-18              ! atomic unit of alpha 
      Eau= 4.35974434D-18                  ! atomic unit of energy

      E=hbar*w/Eau   ! photon energy in a.u.

      tmp=0.d0
      do q=6,40     
      tmp=tmp+d1(q)**2*E1(q)*(E1(q)**2-E**2+gam1(q)**2/4)
     &    /((E1(q)**2-E**2+gam1(q)**2/4)**2+(gam1(q)*E)**2) 
     &       +d2(q)**2*E2(q)*(E2(q)**2-E**2+gam2(q)**2/4)
     &    /((E2(q)**2-E**2+gam2(q)**2/4)**2+(gam2(q)*E)**2)        
      end do
      tmp=tmp/3
      alphaS0=tmp*au

C     Add the tail and core contributions
      alphaS0=alphaS0+15.8*au  
   
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      Function alphaS1_R(w)
      implicit none
      integer q
      real*8 w,alphaS1_R,pi,c,hbar,eps0,ea,au,Eau,d1(6:40),d2(6:40),
     & E1(6:40),E2(6:40),gam1(6:40),gam2(6:40),E,SJ1,SJ2,J,JP,tmp
      common /S/ d1,d2,E1,E2,gam1,gam2
      pi=4*datan(1.D0)      
      c=299792458.D0           ! velocity of light in m/s
      hbar=1.054571726D-34     ! reduced Planck constant in J s
      eps0=8.854187817D-12     ! permittivity of free space   
      ea=1.602176565D-19*0.52917721092D-10 ! atomic unit  ea of dipole
	au=ea**2/4.35974434D-18              ! atomic unit of alpha 
      Eau= 4.35974434D-18                  ! atomic unit of energy

      J=1/2.d0
      JP=1/2.d0
      SJ1=(J*(J+1)-JP*(JP+1)+2)/sqrt(24*J*(J+1)*(2*J+1))
      JP=3/2.d0
      SJ2=(J*(J+1)-JP*(JP+1)+2)/sqrt(24*J*(J+1)*(2*J+1))

      E=hbar*w/Eau   ! photon energy in a.u.

      tmp=0.d0
      do q=6,40     
      tmp=tmp+SJ1*d1(q)**2*E*(E1(q)**2-E**2-gam1(q)**2/4)
     &    /((E1(q)**2-E**2+gam1(q)**2/4)**2+(gam1(q)*E)**2)
     &       +SJ2*d2(q)**2*E*(E2(q)**2-E**2-gam2(q)**2/4)
     &    /((E2(q)**2-E**2+gam2(q)**2/4)**2+(gam2(q)*E)**2)     
      end do
      tmp=2*sqrt(3.d0)*tmp
      alphaS1_R=tmp*au
       
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Function alphaP0(w)
      implicit none
      integer q 
      real*8 w,alphaP0,hbar,ea,au,Eau,J,tmp,d1(5:42),d2(5:42),d3(5:42),
     & E1(5:42),E2(5:42),E3(5:42),E,gam1(5:42),gam2(5:42),gam3(5:42)
      common /P/ d1,d2,d3,E1,E2,E3,gam1,gam2,gam3
      hbar=1.054571726D-34     ! reduced Planck constant in J s
      ea=1.602176565D-19*0.52917721092D-10 ! atomic unit  ea of dipole
	au=ea**2/4.35974434D-18              ! atomic unit of alpha 
      Eau= 4.35974434D-18                  ! atomic unit of energy

      J=3/2.d0 
      
      E=hbar*w/Eau   ! photon energy in a.u.

      tmp=0.d0      
      do q=5,42
      tmp=tmp+d1(q)**2*E1(q)*(E1(q)**2-E**2+gam1(q)**2/4)
     &    /((E1(q)**2-E**2+gam1(q)**2/4)**2+(gam1(q)*E)**2)     
     &       +d2(q)**2*E2(q)*(E2(q)**2-E**2+gam2(q)**2/4)
     &    /((E2(q)**2-E**2+gam2(q)**2/4)**2+(gam2(q)*E)**2)     
     &       +d3(q)**2*E3(q)*(E3(q)**2-E**2+gam3(q)**2/4)
     &    /((E3(q)**2-E**2+gam3(q)**2/4)**2+(gam3(q)*E)**2)          
      end do
      tmp=tmp*2/(3*(2*J+1))
	alphaP0=tmp*au	 

C     Add the tail and core contributions
      alphaP0=alphaP0+15.8*au  
					     	 
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Function alphaP1_R(w)
      implicit none
      integer q 
      real*8 w,alphaP1_R,hbar,ea,au,Eau,J,JP,tmp,d1(5:42),d2(5:42),
     & d3(5:42),E1(5:42),E2(5:42),E3(5:42),E,gam1(5:42),gam2(5:42),
     & gam3(5:42),SJ1,SJ2,SJ3 
      common /P/ d1,d2,d3,E1,E2,E3,gam1,gam2,gam3
      hbar=1.054571726D-34     ! reduced Planck constant in J s
      ea=1.602176565D-19*0.52917721092D-10 ! atomic unit  ea of dipole
	au=ea**2/4.35974434D-18              ! atomic unit of alpha 
      Eau= 4.35974434D-18                  ! atomic unit of energy
		
      E=hbar*w/Eau   ! photon energy in a.u.
 
      J=3/2.d0 
	JP=1/2.d0
	SJ1=(J*(J+1)-JP*(JP+1)+2)/sqrt(J*(J+1)*(2*J+1))
	JP=3/2.d0
	SJ2=(J*(J+1)-JP*(JP+1)+2)/sqrt(J*(J+1)*(2*J+1))
	JP=5/2.d0
	SJ3=(J*(J+1)-JP*(JP+1)+2)/sqrt(J*(J+1)*(2*J+1))

      tmp=0.d0      
      do q=5,42
      tmp=tmp+SJ1*d1(q)**2*E*(E1(q)**2-E**2-gam1(q)**2/4)
     &    /((E1(q)**2-E**2+gam1(q)**2/4)**2+(gam1(q)*E)**2)
     &       +SJ2*d2(q)**2*E*(E2(q)**2-E**2-gam2(q)**2/4)
     &    /((E2(q)**2-E**2+gam2(q)**2/4)**2+(gam2(q)*E)**2)     
     &       +SJ3*d3(q)**2*E*(E3(q)**2-E**2-gam3(q)**2/4)
     &    /((E3(q)**2-E**2+gam3(q)**2/4)**2+(gam3(q)*E)**2)         
      end do
      tmp=tmp/sqrt(2.d0)
	alphaP1_R=tmp*au	 
	  
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Function alphaP2_R(w)
      implicit none
      integer q 
      real*8 w,alphaP2_R,hbar,ea,au,Eau,J,tmp,d1(5:42),d2(5:42),
     & d3(5:42),E1(5:42),E2(5:42),E3(5:42),E,gam1(5:42),gam2(5:42),
     & gam3(5:42),SJ1,SJ2,SJ3 
      common /P/ d1,d2,d3,E1,E2,E3,gam1,gam2,gam3
      hbar=1.054571726D-34     ! reduced Planck constant in J s
      ea=1.602176565D-19*0.52917721092D-10 ! atomic unit  ea of dipole
	au=ea**2/4.35974434D-18              ! atomic unit of alpha 
      Eau= 4.35974434D-18                  ! atomic unit of energy

      E=hbar*w/Eau   ! photon energy in a.u.
 
      J=3/2.d0 
      
      SJ1=-1/(2*sqrt(6.))     
	SJ2=sqrt(2/3.)/5      
	SJ3=-1/(10*sqrt(6.))
	
      tmp=0.d0      
      do q=5,42
      tmp=tmp+SJ1*d1(q)**2*E1(q)*(E1(q)**2-E**2+gam1(q)**2/4)
     &    /((E1(q)**2-E**2+gam1(q)**2/4)**2+(gam1(q)*E)**2)     
     &       +SJ2*d2(q)**2*E2(q)*(E2(q)**2-E**2+gam2(q)**2/4)
     &    /((E2(q)**2-E**2+gam2(q)**2/4)**2+(gam2(q)*E)**2)          
     &       +SJ3*d3(q)**2*E3(q)*(E3(q)**2-E**2+gam3(q)**2/4)
     &    /((E3(q)**2-E**2+gam3(q)**2/4)**2+(gam3(q)*E)**2)          
      end do
      tmp=4*sqrt(5*J*(2*J-1)/(6*(J+1)*(2*J+1)*(2*J+3)))*tmp
      tmp=-0.5*sqrt(6*(J+1)*(2*J+1)*(2*J+3)/(J*(2*J-1)))*tmp         
	alphaP2_R=tmp*au	 
		   
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      Subroutine Magic
      implicit none
      integer FS,FP
      real*8 pi,c,ea,au,lam,w,alphaS0,alphaS1_R,alphaP0,alphaP1_R,
     & alphaP2_R, tmp,mag,cS1,cP1,cP2,ccS1(3:4),ccP1(2:5),ccP2(2:5),J
      character*8 basis
      external alphaS0,alphaS1_R,alphaP0,alphaP1_R,alphaP2_R
      pi=4*datan(1.D0)
      c=299792458.D0           ! velocity of light in m/s
      ea=1.602176565D-19*0.52917721092D-10 ! atomic unit  ea of dipole
	au=ea**2/4.35974434D-18              ! atomic unit of alpha 
	
      mag=40000
      
      basis='J'                ! J or F
      FS=4                     ! 3 or 4 
      FP=5                     ! 2,3,4,5
      
      if (basis.eq.'J') then
      J=1/2.d0
      cS1=-sqrt(2*J/((J+1)*(2*J+1)))     
      J=3/2.d0
      cP1= -sqrt(2*J/((J+1)*(2*J+1)))
      cP2= -2*sqrt(J*(2*J-1)/(6*(J+1)*(2*J+1)*(2*J+3)))
      end if
      
      if (basis.eq.'F') then
      ccS1(3)=sqrt(3.d0)/4
      ccS1(4)=-1/sqrt(3.d0)
      cS1=ccS1(FS)         
      ccP1(2)=sqrt(2.d0/15)
      ccP1(3)=0.d0
      ccP1(4)= -4*sqrt(2.d0/15)/5
      ccP1(5)= -sqrt(3.d0/10)      
      cP1=ccP1(FP)     
      ccP2(2)= -sqrt(2.d0/15)/7    
      ccP2(3)= sqrt(5.d0/6)/6    
      ccP2(4)= sqrt(2.d0/15)/5    
      ccP2(5)= -1/sqrt(30.d0)     
      cP2=ccP2(FP)     
      end if

C     Test
      lam=1064.D0               ! in nm
      w=2*pi*c/(lam*1.D-9)
      write(*,*) 'Estimate at',real(lam),' nm'
      write(*,*) 'For the ground state: aS0 and aS1' 
      write(*,*) alphaS0(w)/au,cS1*alphaS1_R(w)/au
      write(*,*) 'For the excited state: aP0, aP1, aP2' 
      write(*,*) alphaP0(w)/au,cP1*alphaP1_R(w)/au,cP2*alphaP2_R(w)/au
C     End test

      open(1,file='aS0.dat')
      open(2,file='aS1.dat')
      open(3,file='aP0.dat')
      open(4,file='aP1.dat')
      open(5,file='aP2.dat')

      lam=500.D0               ! in nm
      do while (lam.le.8000.D0)
      w=2*pi*c/(lam*1.D-9)
      
      tmp=alphaS0(w)/au
      if (abs(tmp).le.mag) then
      write(1,*) lam,tmp
      end if      
      tmp=cS1*alphaS1_R(w)/au
      if (abs(tmp).le.mag) then
      write(2,*) lam,tmp
      end if          
      tmp=alphaP0(w)/au
      if (abs(tmp).le.mag) then
      write(3,*) lam,tmp
      end if
      tmp=cP1*alphaP1_R(w)/au
      if (abs(tmp).le.mag) then
      write(4,*) lam,tmp
      end if
      tmp=cP2*alphaP2_R(w)/au
      if (abs(tmp).le.mag) then
      write(5,*) lam,tmp
      end if      
      
      lam=lam+1.D-1
      end do

      close(1);close(2);close(3);close(4);close(5)      
	return  
	End

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


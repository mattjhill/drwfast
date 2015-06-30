    subroutine trimult(a,b,c,x,y,n)

!  Does the multipli!ation of a vector by a tridiagonal matrix, i.e.,
!
!      y(i) = a(i)*x(i-1) + b(i)*x(i) + c(i)*x(i+1), i=1,...,n,
!
!   with special cases i=1 and i=n, for which the terms containing
!   a(1) and c(n) are assumed absent.    G. Rybicki

        integer n
        double precision, intent(in) :: a(n-1)
        double precision, intent(in) :: b(n)
        double precision, intent(in) :: c(n-1)
        double precision, intent(in) :: x(n)
        double precision, intent(out) :: y(n)
        integer i

        y(1) = b(1)*x(1) + c(1)*x(2)
        do i=2,n-1
            y(i) = a(i-1)*x(i-1) + b(i)*x(i) + c(i)*x(i+1)
        end do
        y(n) = a(n-1)*x(n-1) + b(n)*x(n)

    end subroutine trimult

    subroutine trisolve(a,b,c,x,y,n)

!   Solves the tridiagonal system
!   
!       a(i)*x(i-1)+b(i)*x(i)+c(i)*x(i+1)=y(i),   i=1,n
!   
!   where the terms containing a(1) and c(n) do not appear.
!   Arrays d and e are scratch arrays of dimension NDIM, which 
!   must be at least equal to n-1.   G. Rybicki

        integer :: n
        double precision, intent(in) :: a(n-1)
        double precision, intent(in) :: b(n)
        double precision, intent(in) :: c(n-1)
        double precision, intent(out) :: x(n)
        double precision, intent(in) :: y(n)
        double precision bb, d(n), e(n)
        integer i, k, n1

        n1=n-1
        d(1)=-c(1)/b(1)
        e(1)=y(1)/b(1)
        do i=2,n1
            bb=b(i)+a(i-1)*d(i-1)
            d(i)=-c(i)/bb
            e(i)=(y(i)-a(i-1)*e(i-1))/bb
        enddo
        x(n)=(y(n)-a(n1)*e(n1))/(b(n)+a(i-1)*d(n1))
        do k=n1,1,-1
            x(k)=e(k)+d(k)*x(k+1)
        enddo

    end subroutine trisolve

    subroutine tridet(a,b,c,n,logdet,signdet)

!   Calculate the determinant of the tridiagonal matrix with diagonals
!   a, b, and c.  The sign is provided by signdet and the magnitude is
!   provided by its logarithm logdet.  The elements a(1) and c(n)
!   are irrelevant and are not used.    G. Rybicki
    
        integer n
        double precision, intent(in) :: a(n-1)
        double precision, intent(in) :: b(n)
        double precision, intent(in) :: c(n-1)
        double precision, intent(out) :: logdet
        double precision, intent(out) :: signdet
        double precision bb,dd,ONE
        integer i, n1
        parameter (ONE=1.d0)

        n1=n-1
        signdet=sign(ONE,b(1))
        logdet=log(abs(b(1)))
        dd=-c(1)/b(1)
        do i=2,n1
            bb=b(i)+a(i-1)*dd
            dd=-c(i)/bb
            signdet=signdet*sign(ONE,bb)
            logdet=logdet+log(abs(bb))
        enddo
        bb=b(n)+a(n1)*dd
        signdet=signdet*sign(ONE,bb)
        logdet=logdet+log(abs(bb))

    end subroutine tridet

    subroutine snsolve(n,t,err2,var,tcorr,ldetc,x,y)
! 
!       Solves the linear system (S+N)*x=y, where S is a (signal)
!       autocorrelation matrix with elements of the form,
!   
!           S(i,j)=var*exp(-abs(t(i)-t(j))/tcorr),  i,j=1,...,n
!   
!       where t(i), i=1,...,n, are an given array of "times", and
!       where N is a diagonal (noise) matrix with elements
!   
!           N(i,j)=delta(i,j)*err2(i)
!   
!       Given a right hand side vector, y(i), i=1,...,n, the unknown
!       vector x(i), i=1,...,n, is found.  Also calculated is ldetc,
!       the logarithm of the determinant of C=S+N.  
!   
!       The integer parameter "mode" must be set to 0 for the first call
!       to the routine for a given set of input parameters n,t,err2,var,tcorr.
!       Subsequent calls with these same parameters, but with different
!       RHSs y(i), can be done with mode.ne.0, which avoids unnecessary
!       computations, increasing the speed.
!   
!       The fast method of Rybicki and Press (1995, Phys. Rev. Lett., 
!       74, 1060; cf.eq. [17]) is used.  
!   
!       This routine assumes that array t(i) is sorted in ascending order,
!       and that there are no duplicate times, since this makes the matrix 
!       S singular.
!   
!       G. Rybicki.  Version 24 April 1998.
!   
!       USES: trimult,tridiag,tridet  (in package subs.f)
!
        integer :: n
        double precision, intent(in) :: t(n), err2(n), var, tcorr, y(n)
        double precision, intent(out) :: ldetc, x(n)
        double precision r(n-1), e(n-1), z(n), ldetmat, ldets, ldetm
        double precision a(n-1), b(n), c(n-1)
        double precision aa(n-1), bb(n), cc(n-1)
        double precision varinv, ZERO, ONE, dum
        integer i, n1
        parameter (ZERO=0.d0,ONE=1.d0)

        if(n.eq.1)then
            x(1)=y(1)/(var+err2(1))
            ldetc=log(var+err2(1))
            return
        endif
        
        n1=n-1
!
!               Check if times are properly sorted.
!
        do i=1,n1
                if(t(i+1).lt.t(i))then
        stop 'snfast11: Array t(i) not properly sorted'
                endif
        enddo

        varinv=ONE/var
!
!           Calculate log-determinant of S using special formulas
!           for Ornstein-Unlenbeck process.
!
        ldets=log(var)
        do i=1,n1
            r(i)=exp(-(t(i+1)-t(i))/tcorr)
            e(i)=r(i)/(ONE-r(i)**2)
            ldets=ldets+log(var*(ONE-r(i)**2))
        enddo
!
!           Calculate the arrays a, b, and c, which define the 
!           tridiagonal matrix inverse S^{-1}.
!
        b(1)= varinv*(ONE+r(1)*e(1))
        c(1)=-varinv*e(1)
        do i=2,n1
            a(i-1)=-varinv*e(i-1)
            b(i)= varinv*(ONE+r(i)*e(i)+r(i-1)*e(i-1))
            c(i)=-varinv*e(i)
        enddo
        a(n1)=-varinv*e(n1)
        b(n)= varinv*(ONE+r(n1)*e(n1))
!
!           Calcutate the arrays aa, bb, and cc, which define the
!           tridiagonal matrix (1+N*S^{-1})
!
        bb(1)=ONE+b(1)*err2(1)
        cc(1)=c(1)*err2(1)
        do i=2,n1
            aa(i-1)=a(i-1)*err2(i)
            bb(i)=ONE+b(i)*err2(i)
            cc(i)=c(i)*err2(i)
        enddo
        aa(n1)=a(n1)*err2(n)
        bb(n)=ONE+b(n)*err2(n)
!
!           Calculate log-determinant of (1+N*S^{-1}).  Finally 
!           get ldetmat, the log-determinant of the full correlation
!           matrix C=S+N=(1+N*S^{-1})*S
!
        call tridet(aa,bb,cc,n,ldetm,dum)
        ldetmat=ldets+ldetm     
!

!
!       Use the quantities determined above to solve system (S+N)x=y
!       for x, given y.  Note that the calculation involves only simple
!       tridiagonal matrices.
!
        call trisolve(aa,bb,cc,z,y,n)
        call trimult(a,b,c,z,x,n)
        ldetc=ldetmat
        return
    
    end subroutine snsolve


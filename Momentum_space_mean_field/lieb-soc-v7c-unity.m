(* ::Package:: *)

(* ::Text::Initialization:: *)
(*v2: updated version following notes, plus Mean-field*)
(*v3: mean-field (more) + \[Epsilon] implementation, bsp*)


(* ::Text::Initialization:: *)
(*v4: mean-field corrected*)


(* ::Text::Initialization:: *)
(*v5: mean-field zero solutions*)
(*v5-check: check why zero solutions*)
(*v5-ch-mu: do MF at a fixed \[Mu]*)
(*v6: follow notes*)
(*v7: clean-up, ZA: n vs \[Mu]*)
(*v7c: clean notebook*)


(* ::Subsection::Initialization::Closed:: *)
(*Set Options*)


(* ::Text::Initialization:: *)
(*Automating good looking plots*)


(* ::Input::Initialization:: *)
SetOptions[ DiscretePlot,Joined-> True, Filling-> None,Frame-> True,LabelStyle->{FontSize->18,FontFamily->"Times",Black},ImageSize->Medium];
SetOptions[ ListPlot,Joined-> True, Filling-> None,Frame-> True,LabelStyle->{FontSize->18,FontFamily->"Times",Black},ImageSize->Medium,PlotStyle->{Black,Thick}];


(* ::Input::Initialization:: *)
SetOptions[ ListContourPlot,Frame-> True,LabelStyle->{FontSize->18,FontFamily->"Times",Black},ImageSize->Medium]; SetOptions[ ListDensityPlot,Frame-> True,LabelStyle->{FontSize->18,FontFamily->"Times",Black},ImageSize->Medium]; 


(* ::Input::Initialization:: *)
SetOptions[ ContourPlot,Frame-> True,LabelStyle->{FontSize->18,FontFamily->"Times",Black},ImageSize->Medium]; SetOptions[ DensityPlot,Frame-> True,LabelStyle->{FontFamily->"CMU Serif",FontSize->24,Black},ImageSize->Medium]; 


(* ::Subsection::Initialization::Closed:: *)
(*Parameters*)


(* ::Text::Initialization:: *)
(*Lattice vectors, and basis vectors*)


(* ::Input::Initialization:: *)
a1 = {1,0};a2 ={0,1};\[Tau]A = {1/2,0};\[Tau]B = {0,0}; \[Tau]C = {0,1/2};\[Tau]vec = {\[Tau]A,\[Tau]B, \[Tau]C};


(* ::Text::Initialization:: *)
(*Reciprocal lattice vectors*)


(* ::Input::Initialization:: *)
LatMat = {a1,a2}; 
ReciLatMat = 2\[Pi] Inverse[LatMat]\[ConjugateTranspose];
b1 = ReciLatMat[[1]] ;b2 = ReciLatMat[[2]];


(* ::Input::Initialization:: *)
\[Delta]f=N[1/(2\[Sqrt]2)];


(* ::Text::Initialization:: *)
(*kSpan : set of points for k-sums, and rSpan : set of points for r-sums*)


(* ::Input::Initialization:: *)
kSpan[krange_]:=N[Flatten[Table[i*b1+j*b2,{i,0,0.999,1/krange},{j,0,0.999,1/krange}],1]];


(* ::Input::Initialization:: *)
rSpan[range_]:=N[Flatten[Table[i*a1+j*a2,{i,-range,range},{j,-range,range}],1]];


(* ::Input::Initialization:: *)
inp[a_,b_]:= Conjugate[a] . b;


(* ::Text::Initialization:: *)
(*We will play around with k-range and range to check for convergence*)


(* ::Input::Initialization:: *)
Fermi[\[Beta]_,x_]:=Chop[N[ 1/(1+ Exp[ \[Beta] x])]];DFermi[\[Beta]_,x_] :=D[ Fermi[\[Beta],x1],x1]/.{x1-> x};


(* ::Subsection::Initialization::Closed:: *)
(*Bloch Hamiltonian*)


(* ::Input::Initialization:: *)
MatX[\[Lambda]_,\[Delta]_,k_] := Module[ {\[Lambda]R = \[Lambda][[1]], \[Lambda]D = \[Lambda][[2]],\[Delta]x = \[Delta][[1]], \[Delta]y = \[Delta][[2]]}, (1+\[Delta]x)(IdentityMatrix[2]+ I \[Lambda]R PauliMatrix[2] +I \[Lambda]D PauliMatrix[1]) + (1-\[Delta]x)Exp[I k . a1](IdentityMatrix[2]- I \[Lambda]R PauliMatrix[2] -I \[Lambda]D PauliMatrix[1])];


(* ::Input::Initialization:: *)
MatY[\[Lambda]_,\[Delta]_,k_] := Module[ {\[Lambda]R = \[Lambda][[1]], \[Lambda]D = \[Lambda][[2]],\[Delta]x = \[Delta][[1]], \[Delta]y = \[Delta][[2]]}, (1+\[Delta]y)(IdentityMatrix[2]- I \[Lambda]R PauliMatrix[1] +I \[Lambda]D PauliMatrix[2]) + (1-\[Delta]y)Exp[I k . a2](IdentityMatrix[2]+ I \[Lambda]R PauliMatrix[1] -I \[Lambda]D PauliMatrix[2])];


(* ::Input::Initialization:: *)
Clear[hamLieb];hamLieb[bsp_,k_] := hamLieb[bsp,k] = Module[{\[Lambda] = bsp[[1]], \[Delta] = bsp[[2]], \[Epsilon] = bsp[[3]]},Module[ {mx = MatX[\[Lambda],\[Delta],k],my = MatY[\[Lambda],\[Delta],k] },ArrayFlatten[ {{-\[Epsilon] IdentityMatrix[2],mx,my},{mx\[ConjugateTranspose],0,0},{my\[ConjugateTranspose],0,0}} ]]];


(* ::Input::Initialization:: *)
Clear[EnerLieb];EnerLieb[bsp_,k_] := EnerLieb[bsp,k]= Chop[  N[Sort[Eigenvalues[hamLieb[bsp,k] ]]] ];
Clear[ULieb];ULieb[bsp_,k_] :=ULieb[bsp,k]=Transpose[Chop[N[   Transpose[ SortBy[ Transpose[ Eigensystem[ hamLieb[bsp,k] ] ] ,First ] ][[2]] ] ] ] ;


(* ::Subsubsection::Initialization::Closed:: *)
(*Plot Band Structure*)


(* ::Input::Initialization:: *)
Gama = {0,0}; Xpoint = b1/2; Mpoint = 1/2 (b1+b2);


(* ::Input::Initialization:: *)
BetAB[ a_, b_] = Table[ (1-i)*a +i*b , {i,0.,1,0.01}];


(* ::Input::Initialization:: *)
kCut = Join[BetAB[ Gama,Xpoint]  ,BetAB[Xpoint,Mpoint], BetAB[Mpoint,Gama] ];


(* ::Input::Initialization:: *)
PlotDispData[bsp_] := Table[ EnerLieb[bsp,k]  ,{k,kCut} ];


(* ::Input::Initialization:: *)
Block[ {bsp={{0.1,0.1},{0.2,0.2},0.0}},ListPlot[Table[ PlotDispData[bsp][[All,i]],{i,1,6}],PlotStyle->Black,
FrameTicks->{{{1,"\[CapitalGamma]"},{101,"X"},{202,"M"},{303,"\[CapitalGamma]"}},Automatic},Joined->True,FrameLabel->{None,"E/t"},AspectRatio->1/1]]


(* ::Input::Initialization:: *)
Block[{bsp={{0.45,0.0},{0.0,0.0},0.0},k=RandomReal[1,2]}, EnerLieb[bsp,k]]


(* ::Subsection::Initialization::Closed:: *)
(*Mean Field Theory*)


(* ::Subsubsection::Initialization:: *)
(*Mean-Field self-consistent set up*)


(* ::Text::Initialization:: *)
(*Here nup, ndn, splus and sdown are all 3 vectors (for A, B, C)*)


(* ::Input::Initialization:: *)
IJMat[i_,j_]:= ArrayFlatten[ Table[ If[i1==i&&j1==j,1,0],{i1,1,6},{j1,1,6}]];


(* ::Text::Initialization:: *)
(*MF vec = { Subscript[n, A\[UpArrow]], Subscript[n, A\[DownArrow]], Subscript[n, B\[UpArrow]],  Subscript[n, B\[DownArrow]], Subscript[n, C\[UpArrow]],Subscript[n, C\[DownArrow]], Subscript[SuperPlus[S], A], Subscript[SuperPlus[S], B], Subscript[SuperPlus[S], C]}*)


(* ::Input::Initialization:: *)
MvMat = {IJMat[1,1],IJMat[2,2],IJMat[3,3],IJMat[4,4],IJMat[5,5],IJMat[6,6],IJMat[1,2],IJMat[3,4],IJMat[5,6]};


(* ::Input::Initialization:: *)
MFHamvec = {IJMat[2,2],IJMat[1,1],IJMat[4,4],IJMat[3,3],IJMat[6,6],IJMat[5,5],-IJMat[2,1],-IJMat[4,3],-IJMat[6,5]};


(* ::Input::Initialization:: *)
Clear[HamMFU];HamMFU[MFvec_]:= HamMFU[MFvec] =MFvec[[1;;6]] . MFHamvec[[1;;6]] +MFvec[[7;;9]] . MFHamvec[[7;;9]]+(MFvec[[7;;9]] . MFHamvec[[7;;9]])\[ConjugateTranspose];


(* ::Input::Initialization:: *)
Clear[HamMFull];HamMFull[bsp_,MFvec_,k_,U_,\[Mu]_]:= HamMFull[bsp,MFvec,k,U,\[Mu]]= hamLieb[bsp,k] +U HamMFU[MFvec]- (\[Mu]+U/2)IdentityMatrix[6];


(* ::Input::Initialization:: *)
HamMFull[{{0.0,0.0},{0.2,0.2},0.0},{nAu, nAd, nBu, nBd, nCu, nCd, SpA, SpB, SpC},Gama,U,\[Mu]]//MatrixForm


(* ::Input::Initialization:: *)
Clear[EnerMF];EnerMF[bsp_,MFvec_,k_,U_,\[Mu]_] := EnerMF[bsp,MFvec,k,U,\[Mu]]= Chop[  N[Sort[Eigenvalues[HamMFull[bsp,MFvec,k,U,\[Mu]] ]]] ];


(* ::Input::Initialization:: *)
Clear[ULiebMF];ULiebMF[bsp_,MFvec_,k_,U_,\[Mu]_] :=ULiebMF[bsp,MFvec,k,U,\[Mu]]=Transpose[Transpose[SortBy[Transpose[Eigensystem[ HamMFull[bsp,MFvec,k,U,\[Mu]]]],First]][[2]]]


(* ::Input::Initialization:: *)
Clear[MFvalue];MFvalue[bsp_,MFvec_,U_,\[Mu]_,\[Beta]_,krange_] :=MFvalue[bsp,MFvec,U,\[Mu],\[Beta],krange]=Table[1/Length[kSpan[krange]] Chop[Sum[ Module[{um = ULiebMF[bsp,MFvec,k,U,\[Mu]]}, Fermi[\[Beta], EnerMF[bsp,MFvec,k,U,\[Mu]][[band]] ](um\[ConjugateTranspose] . MvMat[[mfvectors]] . um)[[band,band]]],{band,1 ,6},{k,kSpan[krange]}]],{mfvectors,1,9}];


(* ::Text::Initialization:: *)
(*Take the initial seed from the U \[RightArrow] \[Infinity] limit*)


(* ::Input::Initialization:: *)
InitSeed = Table[ 0.5, {i,1,9}];


(* ::Input::Initialization:: *)
Clear[MFsoln];MFsoln[bsp_,U_,\[Mu]_,\[Beta]_,krange_]:= MFsoln[bsp,U,\[Mu],\[Beta],krange]=MFvec/.FindRoot[ MFvec - MFvalue[bsp,MFvec,U,\[Mu],\[Beta],krange],{MFvec,InitSeed},Evaluated->False,MaxIterations->10,PrecisionGoal->3,AccuracyGoal->3]


(* ::Input::Initialization:: *)
Clear[NumEq];NumEq[MFvec_] :=NumEq[MFvec]=Total[ MFvec[[1;;6]] ];


(* ::Subsection::Initialization::Closed:: *)
(*Half-filling, \[Mu]=0*)


(* ::Input::Initialization:: *)
Clear[MagnHF];MagnHF[bsp_,U_, \[Beta]_,krange_,orb_]:=  MagnHF[bsp,U,\[Beta],krange,orb]=Module[{itrMF=MFsoln[bsp,U,0.0,\[Beta],krange]},{Re[itrMF[[orb+6]]] ,Im[itrMF[[orb+6]]], 1/2 ( itrMF[[orb+3]] - itrMF[[orb]])}];


(* ::Subsection::Initialization:: *)
(*Export files*)


(* ::Input::Initialization:: *)
dataHF = Block[ {bsp={{0.0,0.0},{0.2,0.2},0.0}, \[Beta]=100, krange=30},Table[ArrayFlatten[{ U,MFsoln[bsp,U,0.0,\[Beta],krange]},1], {U,0.0,10.0}]];


(* ::Input::Initialization:: *)
Export["/home/verma.148/fbint/dataHF.txt",dataHF];


(* ::Input::Initialization:: *)
dataU0p2mu = Block[ {bsp={{0.0,0.0},{0.2,0.2},0.0},U=0.2, \[Beta]=100, krange=30},Table[ArrayFlatten[{ \[Mu],MFsoln[bsp,U,\[Mu],\[Beta],krange]},1], {\[Mu],-0.1,0.1,0.01}]];


(* ::Input::Initialization:: *)
Export["/home/verma.148/fbint/dataU0p2mu.txt",dataU0p2mu];


(* ::Input::Initialization:: *)
dataU2p0mu = Block[ {bsp={{0.0,0.0},{0.2,0.2},0.0},U=2.0, \[Beta]=100, krange=30},Table[ArrayFlatten[{ \[Mu],MFsoln[bsp,U,\[Mu],\[Beta],krange]},1], {\[Mu],-0.1,0.1,0.01}]];


(* ::Input::Initialization:: *)
Export["/home/verma.148/fbint/dataU2p0mu.txt",dataU2p0mu];

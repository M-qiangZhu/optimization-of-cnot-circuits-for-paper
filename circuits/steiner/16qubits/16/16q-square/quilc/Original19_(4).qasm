// EXPECTED_REWIRING [1 7 2 3 9 4 5 0 8 13 10 11 12 14 6 15]
// CURRENT_REWIRING [3 8 1 11 10 5 2 0 7 9 13 4 12 14 6 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.7843290499389812*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.8103884456122044*pi) q[2];
rz(0.59368010174542*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.958108965734335*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.4741817804854853*pi) q[7];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[13];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(-1.784329049938982*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.0636030200628972*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.760407881182692*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(0.10344064106915161*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.3572636036508121*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.077989633526896*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
cz q[10], q[9];
rz(0.59368010174542*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.958108965734335*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.4741817804854853*pi) q[3];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.164582056715158*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687736*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[15];
rz(-3.0381520125206416*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.7843290499389812*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268964*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.381184772407101*pi) q[14];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(3.141592653589793*pi) q[13];
rz(1.4564375502462918*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.4269954866939927*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(0.08197635545524928*pi) q[14];
rz(2.487346972302436*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.217366411692773*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
cz q[14], q[13];
rz(1.9770105968746368*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.9762070474829154*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.431535963529496*pi) q[5];
cz q[10], q[5];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.077989633526896*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.8103884456122044*pi) q[13];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.6013875941795643*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.818218467743224*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.8722426477468238*pi) q[9];
rz(-2.386974844031116*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.9045571192789916*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[9];
rz(0.6712412220861737*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(2.1644764285403166*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.958108965734335*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[13], q[10];
rz(2.238207199899204*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[12];
cz q[11], q[12];
rz(1.9462014062908715*pi) q[5];
rx(3.141592653589793*pi) q[5];
rz(1.550179670184607*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.2678395722928435*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.06900379437231875*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[4];
rz(1.3572636036508117*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.077989633526896*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-2.7926816769436824*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.6468601754530754*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.06290232120514361*pi) q[9];
rz(2.814439955788636*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.8204413152673096*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rx(-1.5707963267948966*pi) q[9];
rz(-1.435978424752701*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[9], q[10];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(-1.1645820567151595*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.1653856061068779*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.5146654427613733*pi) q[11];
rz(1.2303146718387175*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.045286428791384*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rx(1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[12];
rz(2.786122328771262*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.7211841762790168*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.4909137555525467*pi) q[9];
rz(-2.4023464166981015*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[12], q[11];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(0.6463506146747173*pi) q[1];
rz(-2.4952420389150767*pi) q[2];
rz(-1.1645820567151595*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.1653856061068779*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.5146654427613733*pi) q[3];
rz(2.217146941469614*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.6463506146747164*pi) q[7];
rz(1.0561308840335222*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(3.141592653589793*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(-1.1645820567151595*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.1653856061068779*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5146654427613733*pi) q[13];
rz(0.3742623732732058*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.7340715749037504*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.748399031120085*pi) q[14];
rz(3.141592653589793*pi) q[15];

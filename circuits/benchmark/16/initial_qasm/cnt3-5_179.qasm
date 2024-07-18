OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
cx q[1],q[3];
h q[1];
cx q[4],q[6];
h q[4];
cx q[7],q[9];
h q[7];
cx q[10],q[12];
h q[10];
cx q[13],q[16];
h q[13];
cx q[14],q[13];
tdg q[13];
cx q[15],q[13];
t q[13];
cx q[14],q[13];
tdg q[13];
t q[14];
p(3*pi/4) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(pi) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(11*pi/4) q[14];
cx q[15],q[13];
t q[13];
h q[13];
cx q[13],q[16];
cx q[13],q[10];
tdg q[10];
cx q[11],q[10];
t q[10];
cx q[13],q[10];
tdg q[10];
cx q[11],q[10];
t q[10];
h q[10];
cx q[10],q[12];
cx q[10],q[7];
tdg q[7];
cx q[8],q[7];
t q[7];
cx q[10],q[7];
tdg q[7];
cx q[8],q[7];
t q[7];
h q[7];
cx q[7],q[9];
cx q[7],q[4];
tdg q[4];
cx q[5],q[4];
t q[4];
cx q[7],q[4];
tdg q[4];
cx q[5],q[4];
t q[4];
h q[4];
cx q[4],q[6];
cx q[4],q[1];
tdg q[1];
cx q[2],q[1];
t q[1];
cx q[4],q[1];
tdg q[1];
cx q[2],q[1];
t q[1];
h q[1];
cx q[1],q[3];
p(-0.37528932002377235) q[2];
sdg q[2];
h q[2];
sdg q[2];
p(2*pi) q[2];
sdg q[2];
h q[2];
sdg q[2];
p(7.47869231395071) q[2];
t q[4];
p(-pi) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(4.415354682941693) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(7*pi/2) q[4];
cx q[2],q[4];
p(2.2735405153160766) q[2];
sdg q[2];
h q[2];
sdg q[2];
p(2*pi) q[2];
sdg q[2];
h q[2];
sdg q[2];
p(6.985929495700767) q[2];
p(1.3616152868139437) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(4.630325291773653) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(7.108365232300896) q[4];
cx q[2],q[4];
p(-pi/4) q[2];
sdg q[2];
h q[2];
sdg q[2];
p(3*pi/2) q[2];
sdg q[2];
h q[2];
sdg q[2];
p(11*pi/4) q[2];
cx q[3],q[2];
t q[2];
p(pi/2) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(5.986151009736591) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(5*pi/2) q[4];
cx q[4],q[2];
tdg q[2];
cx q[3],q[2];
t q[2];
h q[2];
p(pi/4) q[3];
sdg q[3];
h q[3];
sdg q[3];
p(3*pi/2) q[3];
sdg q[3];
h q[3];
sdg q[3];
p(5*pi/2) q[3];
t q[4];
p(1.2737620293519) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(3*pi/2) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(2*pi) q[4];
cx q[3],q[4];
p(pi/2) q[3];
sdg q[3];
h q[3];
sdg q[3];
p(3*pi/2) q[3];
sdg q[3];
h q[3];
sdg q[3];
p(15*pi/4) q[3];
p(0) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(3*pi/2) q[4];
sdg q[4];
h q[4];
sdg q[4];
p(8.15101593141748) q[4];
p(-0.37528932002377235) q[5];
sdg q[5];
h q[5];
sdg q[5];
p(2*pi) q[5];
sdg q[5];
h q[5];
sdg q[5];
p(7.47869231395071) q[5];
t q[7];
p(-pi) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(4.415354682941693) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(7*pi/2) q[7];
cx q[5],q[7];
p(2.2735405153160766) q[5];
sdg q[5];
h q[5];
sdg q[5];
p(2*pi) q[5];
sdg q[5];
h q[5];
sdg q[5];
p(6.985929495700767) q[5];
p(1.3616152868139437) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(4.630325291773653) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(7.108365232300896) q[7];
cx q[5],q[7];
p(-pi/4) q[5];
sdg q[5];
h q[5];
sdg q[5];
p(3*pi/2) q[5];
sdg q[5];
h q[5];
sdg q[5];
p(11*pi/4) q[5];
cx q[6],q[5];
t q[5];
p(pi/2) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(5.986151009736591) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(5*pi/2) q[7];
cx q[7],q[5];
tdg q[5];
cx q[6],q[5];
t q[5];
h q[5];
p(pi/4) q[6];
sdg q[6];
h q[6];
sdg q[6];
p(3*pi/2) q[6];
sdg q[6];
h q[6];
sdg q[6];
p(5*pi/2) q[6];
t q[7];
p(1.2737620293519) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(3*pi/2) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(2*pi) q[7];
cx q[6],q[7];
p(pi/2) q[6];
sdg q[6];
h q[6];
sdg q[6];
p(3*pi/2) q[6];
sdg q[6];
h q[6];
sdg q[6];
p(15*pi/4) q[6];
p(0) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(3*pi/2) q[7];
sdg q[7];
h q[7];
sdg q[7];
p(8.15101593141748) q[7];
p(-0.37528932002377235) q[8];
sdg q[8];
h q[8];
sdg q[8];
p(2*pi) q[8];
sdg q[8];
h q[8];
sdg q[8];
p(7.47869231395071) q[8];
t q[10];
p(-pi) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(4.415354682941693) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(7*pi/2) q[10];
cx q[8],q[10];
p(2.2735405153160766) q[8];
sdg q[8];
h q[8];
sdg q[8];
p(2*pi) q[8];
sdg q[8];
h q[8];
sdg q[8];
p(6.985929495700767) q[8];
p(1.3616152868139437) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(4.630325291773653) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(7.108365232300896) q[10];
cx q[8],q[10];
p(-pi/4) q[8];
sdg q[8];
h q[8];
sdg q[8];
p(3*pi/2) q[8];
sdg q[8];
h q[8];
sdg q[8];
p(11*pi/4) q[8];
cx q[9],q[8];
t q[8];
p(pi/2) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(5.986151009736591) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(5*pi/2) q[10];
cx q[10],q[8];
tdg q[8];
cx q[9],q[8];
t q[8];
h q[8];
p(pi/4) q[9];
sdg q[9];
h q[9];
sdg q[9];
p(3*pi/2) q[9];
sdg q[9];
h q[9];
sdg q[9];
p(5*pi/2) q[9];
t q[10];
p(1.2737620293519) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(3*pi/2) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(2*pi) q[10];
cx q[9],q[10];
p(pi/2) q[9];
sdg q[9];
h q[9];
sdg q[9];
p(3*pi/2) q[9];
sdg q[9];
h q[9];
sdg q[9];
p(15*pi/4) q[9];
p(0) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(3*pi/2) q[10];
sdg q[10];
h q[10];
sdg q[10];
p(8.15101593141748) q[10];
p(-0.37528932002377235) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(2*pi) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(7.47869231395071) q[11];
t q[13];
p(-pi) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(4.415354682941693) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(7*pi/2) q[13];
cx q[11],q[13];
p(2.2735405153160766) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(2*pi) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(6.985929495700767) q[11];
p(1.3616152868139437) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(4.630325291773653) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(7.108365232300896) q[13];
cx q[11],q[13];
p(-pi/4) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(3*pi/2) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(11*pi/4) q[11];
cx q[12],q[11];
t q[11];
p(pi/2) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(5.986151009736591) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(5*pi/2) q[13];
cx q[13],q[11];
tdg q[11];
cx q[12],q[11];
t q[11];
h q[11];
p(pi/4) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(3*pi/2) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(5*pi/2) q[12];
t q[13];
p(1.2737620293519) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(3*pi/2) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(2*pi) q[13];
cx q[12],q[13];
p(pi/2) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(3*pi/2) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(15*pi/4) q[12];
p(0) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(3*pi/2) q[13];
sdg q[13];
h q[13];
sdg q[13];
p(8.15101593141748) q[13];
p(-pi) q[15];
sdg q[15];
h q[15];
sdg q[15];
p(5.009423277827686) q[15];
sdg q[15];
h q[15];
sdg q[15];
p(5*pi/2) q[15];
cx q[14],q[15];
p(2.2735405153160766) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(2*pi) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(6.985929495700767) q[14];
p(1.3616152868139437) q[15];
sdg q[15];
h q[15];
sdg q[15];
p(4.630325291773653) q[15];
sdg q[15];
h q[15];
sdg q[15];
p(7.108365232300896) q[15];
cx q[14],q[15];
p(2.0881366445137877) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(2*pi) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(9.942118278488271) q[14];
p(0.7630161060081049) q[15];
sdg q[15];
h q[15];
sdg q[15];
p(4.503922301292532) q[15];
sdg q[15];
h q[15];
sdg q[15];
p(8.852530134710126) q[15];
cx q[16],q[15];
t q[15];
cx q[14],q[15];
t q[14];
p(0.7304668064808832) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(2*pi) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(7.01365211366047) q[14];
tdg q[15];
cx q[16],q[15];
t q[15];
h q[15];
p(-pi/4) q[16];
sdg q[16];
h q[16];
sdg q[16];
p(3.4386269510327896) q[16];
sdg q[16];
h q[16];
sdg q[16];
p(5*pi/2) q[16];
cx q[14],q[16];
p(1.4288992721907325) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(2*pi) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(10.853677232960113) q[14];
p(pi/2) q[16];
sdg q[16];
h q[16];
sdg q[16];
p(5.986151009736591) q[16];
sdg q[16];
h q[16];
sdg q[16];
p(13*pi/4) q[16];

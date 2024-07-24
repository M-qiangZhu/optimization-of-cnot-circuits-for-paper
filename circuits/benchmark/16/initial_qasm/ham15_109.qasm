OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
cx q[10],q[0];
cx q[10],q[5];
cx q[10],q[6];
cx q[10],q[7];
cx q[12],q[8];
cx q[8],q[7];
cx q[8],q[4];
h q[8];
cx q[13],q[11];
cx q[11],q[13];
cx q[12],q[11];
cx q[12],q[8];
tdg q[8];
cx q[11],q[8];
t q[8];
cx q[12],q[8];
tdg q[8];
cx q[11],q[8];
t q[8];
h q[8];
cx q[8],q[4];
cx q[8],q[7];
t q[12];
cx q[11],q[12];
p(5*pi/16) q[11];
tdg q[12];
cx q[11],q[12];
cx q[12],q[3];
h q[12];
cx q[14],q[0];
cx q[14],q[6];
cx q[14],q[8];
cx q[14],q[13];
cx q[14],q[12];
tdg q[12];
cx q[13],q[12];
t q[12];
cx q[14],q[12];
tdg q[12];
cx q[13],q[12];
t q[12];
h q[12];
t q[14];
cx q[13],q[14];
t q[13];
tdg q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[12],q[3];
h q[14];
cx q[13],q[14];
tdg q[14];
cx q[12],q[14];
t q[14];
cx q[13],q[14];
t q[13];
tdg q[14];
cx q[12],q[14];
cx q[12],q[13];
t q[12];
tdg q[13];
cx q[12],q[13];
cx q[12],q[10];
cx q[10],q[8];
cx q[10],q[9];
cx q[10],q[4];
h q[12];
t q[14];
h q[14];
cx q[14],q[12];
tdg q[12];
cx q[13],q[12];
t q[12];
cx q[14],q[12];
tdg q[12];
cx q[13],q[12];
t q[12];
h q[12];
cx q[12],q[10];
cx q[10],q[3];
cx q[10],q[2];
cx q[10],q[1];
h q[10];
t q[14];
cx q[13],q[14];
t q[13];
tdg q[14];
cx q[13],q[14];
p(3*pi/4) q[14];
cx q[14],q[10];
p(-pi/4) q[10];
cx q[14],q[10];
p(pi/4) q[10];
sdg q[14];
h q[14];
sdg q[14];
p(3*pi/4) q[14];
cx q[13],q[14];
p(pi/4) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(pi/2) q[14];
cx q[11],q[14];
p(pi/4) q[14];
cx q[12],q[14];
p(-pi/4) q[14];
cx q[11],q[14];
p(pi/4) q[14];
cx q[12],q[14];
p(pi/4) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(3*pi/4) q[14];
cx q[13],q[14];
p(pi/4) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(3*pi/4) q[14];
cx q[14],q[10];
p(pi/4) q[10];
cx q[14],q[10];
p(-pi/4) q[10];
sdg q[14];
h q[14];
sdg q[14];
p(-5*pi/4) q[14];
cx q[13],q[14];
p(pi/4) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(-5*pi/4) q[14];
cx q[12],q[14];
p(-pi/4) q[14];
cx q[11],q[14];
p(pi/4) q[14];
cx q[12],q[14];
p(-pi/4) q[14];
cx q[11],q[14];
cx q[11],q[10];
p(-pi/16) q[10];
cx q[11],q[10];
p(pi/16) q[10];
cx q[11],q[12];
p(-pi/16) q[12];
cx q[12],q[10];
p(pi/16) q[10];
cx q[12],q[10];
p(-pi/16) q[10];
cx q[11],q[12];
p(3*pi/16) q[12];
cx q[12],q[10];
p(-pi/16) q[10];
cx q[12],q[10];
p(pi/16) q[10];
p(pi/2) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(-5*pi/4) q[14];
cx q[13],q[14];
cx q[12],q[13];
p(-pi/16) q[13];
cx q[13],q[10];
p(pi/16) q[10];
cx q[13],q[10];
p(-pi/16) q[10];
cx q[11],q[13];
p(pi/16) q[13];
cx q[13],q[10];
p(-pi/16) q[10];
cx q[13],q[10];
p(pi/16) q[10];
cx q[12],q[13];
p(-pi/16) q[13];
cx q[13],q[10];
p(pi/16) q[10];
cx q[13],q[10];
p(-pi/16) q[10];
cx q[11],q[13];
p(3*pi/16) q[13];
cx q[13],q[10];
p(-pi/16) q[10];
cx q[13],q[10];
p(pi/16) q[10];
h q[10];
cx q[10],q[9];
cx q[10],q[6];
cx q[10],q[4];
cx q[10],q[3];
cx q[10],q[2];
cx q[10],q[1];
cx q[11],q[10];
cx q[10],q[5];
h q[11];
p(pi/8) q[11];
cx q[12],q[13];
p(-pi/8) q[13];
cx q[12],q[13];
p(pi/4) q[14];
sdg q[14];
h q[14];
sdg q[14];
p(-11*pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[12],q[14];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[12],q[14];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[13],q[11];
p(pi/8) q[11];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[12],q[11];
p(pi/8) q[11];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[13],q[11];
p(pi/8) q[11];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[12],q[11];
h q[11];
cx q[11],q[10];
cx q[12],q[7];
cx q[11],q[12];
h q[11];
p(pi/8) q[11];
p(pi/8) q[12];
cx q[13],q[10];
p(pi/8) q[13];
cx q[12],q[13];
p(-pi/8) q[13];
cx q[12],q[13];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[12],q[14];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[12],q[14];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[13],q[11];
p(pi/8) q[11];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[12],q[11];
p(pi/8) q[11];
sdg q[12];
h q[12];
sdg q[12];
p(5.98615100973659) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(7*pi/2) q[12];
cx q[14],q[11];
p(-pi/8) q[11];
cx q[13],q[11];
p(pi/8) q[11];
cx q[13],q[4];
cx q[13],q[3];
cx q[13],q[2];
h q[13];
p(pi/8) q[13];
cx q[14],q[11];
p(-9*pi/8) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(3*pi/2) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(11.165492742291336) q[11];
cx q[11],q[12];
sdg q[11];
h q[11];
sdg q[11];
p(pi) q[11];
sdg q[11];
h q[11];
sdg q[11];
p(10.825655832837215) q[11];
cx q[11],q[5];
p(pi/2) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(5.98615100973659) q[12];
sdg q[12];
h q[12];
sdg q[12];
p(6.675884388878311) q[12];
cx q[12],q[9];
cx q[11],q[12];
p(-pi/8) q[12];
cx q[11],q[12];
p(pi/8) q[14];
cx q[12],q[14];
p(-pi/8) q[14];
cx q[11],q[14];
p(pi/8) q[14];
cx q[12],q[14];
p(-pi/8) q[14];
cx q[11],q[14];
cx q[14],q[13];
p(-pi/8) q[13];
cx q[12],q[13];
p(pi/8) q[13];
cx q[14],q[13];
p(-pi/8) q[13];
cx q[11],q[13];
p(pi/8) q[13];
cx q[14],q[13];
p(-pi/8) q[13];
cx q[12],q[13];
p(pi/8) q[13];
cx q[14],q[13];
p(-pi/8) q[13];
cx q[11],q[13];
h q[13];
cx q[13],q[10];
cx q[10],q[0];
cx q[10],q[8];
cx q[13],q[3];
cx q[13],q[2];
cx q[12],q[2];
cx q[12],q[0];
h q[12];
cx q[14],q[12];
tdg q[12];
cx q[11],q[12];
t q[12];
cx q[14],q[12];
tdg q[12];
cx q[11],q[12];
t q[12];
h q[12];
cx q[12],q[7];
cx q[12],q[2];
p(pi/8) q[12];
t q[14];
cx q[11],q[14];
tdg q[14];
cx q[11],q[14];
cx q[11],q[12];
p(-pi/8) q[12];
cx q[11],q[12];
cx q[14],q[13];
cx q[13],q[4];
p(pi/8) q[13];
cx q[12],q[13];
p(-pi/8) q[13];
cx q[11],q[13];
p(pi/8) q[13];
cx q[12],q[13];
p(-pi/8) q[13];
cx q[11],q[13];
cx q[14],q[10];
cx q[10],q[7];
cx q[14],q[6];
h q[14];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[12],q[14];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[11],q[14];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[12],q[14];
p(pi/8) q[14];
cx q[13],q[14];
p(-pi/8) q[14];
cx q[11],q[14];
h q[14];
cx q[14],q[12];
cx q[12],q[4];
cx q[12],q[0];
cx q[8],q[12];
cx q[14],q[11];
cx q[14],q[10];
cx q[14],q[6];
h q[14];
cx q[13],q[14];
tdg q[14];
cx q[11],q[14];
t q[14];
cx q[13],q[14];
t q[13];
tdg q[14];
cx q[11],q[14];
cx q[11],q[13];
t q[11];
tdg q[13];
cx q[11],q[13];
t q[14];
h q[14];
cx q[14],q[13];
cx q[14],q[11];
cx q[7],q[11];
cx q[10],q[14];
cx q[13],q[10];
cx q[10],q[13];
cx q[13],q[6];
cx q[6],q[10];
cx q[13],q[9];
cx q[9],q[13];
cx q[9],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[9];
cx q[9],q[5];
cx q[5],q[9];
cx q[5],q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[4],q[7];
cx q[7],q[3];
cx q[3],q[7];
cx q[3],q[6];
cx q[6],q[2];
cx q[2],q[6];
cx q[2],q[5];
cx q[5],q[1];
cx q[1],q[5];
cx q[1],q[4];
cx q[4],q[0];
cx q[0],q[4];
cx q[0],q[3];
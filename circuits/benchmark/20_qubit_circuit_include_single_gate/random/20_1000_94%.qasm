OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[1],q[5];
cx q[9],q[17];
cx q[7],q[6];
t q[8];
cx q[14],q[8];
cx q[13],q[14];
cx q[7],q[8];
cx q[16],q[2];
cx q[2],q[4];
cx q[2],q[9];
cx q[7],q[8];
cx q[4],q[5];
cx q[5],q[19];
cx q[0],q[5];
cx q[5],q[18];
cx q[2],q[11];
cx q[4],q[10];
cx q[0],q[12];
cx q[15],q[19];
cx q[14],q[4];
cx q[11],q[4];
cx q[19],q[10];
x q[17];
cx q[9],q[10];
cx q[10],q[1];
cx q[18],q[13];
cx q[7],q[19];
cx q[8],q[4];
cx q[11],q[1];
cx q[11],q[5];
cx q[7],q[15];
cx q[17],q[19];
cx q[19],q[6];
cx q[18],q[5];
cx q[0],q[13];
cx q[16],q[18];
cx q[14],q[10];
cx q[18],q[8];
cx q[2],q[8];
s q[12];
cx q[3],q[17];
cx q[11],q[7];
cx q[17],q[16];
cx q[0],q[5];
t q[3];
cx q[14],q[5];
cx q[0],q[7];
cx q[17],q[9];
cx q[16],q[9];
cx q[4],q[16];
cx q[9],q[13];
cx q[0],q[17];
cx q[15],q[14];
cx q[5],q[7];
cx q[14],q[12];
cx q[4],q[19];
cx q[18],q[16];
cx q[8],q[4];
cx q[5],q[2];
cx q[17],q[2];
cx q[19],q[10];
cx q[8],q[9];
cx q[12],q[11];
cx q[8],q[19];
cx q[1],q[7];
cx q[1],q[18];
cx q[17],q[18];
cx q[3],q[16];
cx q[6],q[7];
cx q[8],q[18];
cx q[3],q[17];
cx q[18],q[19];
s q[1];
cx q[17],q[0];
cx q[6],q[1];
cx q[11],q[17];
cx q[0],q[9];
cx q[7],q[1];
cx q[5],q[10];
cx q[0],q[3];
cx q[12],q[4];
cx q[15],q[6];
cx q[3],q[19];
cx q[10],q[4];
cx q[7],q[0];
cx q[12],q[6];
x q[6];
cx q[19],q[16];
s q[10];
cx q[16],q[10];
x q[3];
cx q[17],q[12];
cx q[17],q[6];
cx q[4],q[1];
cx q[3],q[17];
cx q[0],q[6];
cx q[9],q[19];
cx q[14],q[6];
cx q[2],q[15];
cx q[5],q[1];
cx q[7],q[6];
cx q[18],q[19];
cx q[4],q[10];
cx q[6],q[5];
cx q[11],q[5];
cx q[19],q[10];
cx q[19],q[16];
cx q[7],q[9];
cx q[14],q[19];
cx q[14],q[15];
cx q[9],q[10];
h q[2];
cx q[4],q[9];
cx q[13],q[16];
cx q[18],q[10];
cx q[13],q[6];
cx q[9],q[19];
cx q[18],q[0];
cx q[0],q[17];
cx q[5],q[7];
cx q[12],q[5];
cx q[5],q[9];
cx q[3],q[19];
cx q[3],q[14];
cx q[16],q[9];
cx q[0],q[9];
cx q[11],q[7];
cx q[15],q[3];
cx q[14],q[16];
cx q[7],q[4];
cx q[19],q[6];
cx q[16],q[18];
cx q[19],q[11];
cx q[7],q[8];
x q[11];
x q[6];
cx q[13],q[11];
cx q[17],q[3];
cx q[14],q[19];
cx q[11],q[15];
cx q[7],q[16];
cx q[12],q[16];
cx q[7],q[19];
cx q[4],q[6];
cx q[7],q[3];
s q[0];
cx q[17],q[0];
cx q[15],q[4];
cx q[1],q[10];
cx q[1],q[13];
cx q[19],q[18];
cx q[3],q[15];
cx q[10],q[14];
cx q[12],q[10];
cx q[19],q[0];
cx q[12],q[13];
cx q[11],q[13];
cx q[18],q[19];
cx q[17],q[19];
cx q[17],q[1];
cx q[12],q[19];
cx q[5],q[9];
cx q[6],q[2];
cx q[4],q[3];
cx q[10],q[12];
cx q[7],q[2];
cx q[2],q[11];
cx q[2],q[10];
cx q[13],q[11];
cx q[6],q[15];
cx q[19],q[13];
cx q[19],q[3];
cx q[16],q[1];
cx q[16],q[19];
cx q[11],q[1];
cx q[1],q[9];
cx q[6],q[10];
cx q[5],q[0];
cx q[11],q[0];
cx q[10],q[8];
cx q[13],q[1];
cx q[4],q[15];
t q[15];
cx q[0],q[1];
h q[4];
t q[5];
cx q[0],q[18];
cx q[12],q[6];
cx q[11],q[10];
cx q[3],q[1];
cx q[8],q[10];
cx q[13],q[6];
cx q[17],q[1];
cx q[15],q[14];
cx q[2],q[13];
cx q[5],q[4];
cx q[18],q[7];
cx q[9],q[5];
cx q[13],q[16];
cx q[13],q[0];
cx q[5],q[6];
cx q[14],q[7];
cx q[5],q[2];
cx q[17],q[3];
cx q[9],q[0];
cx q[19],q[6];
cx q[17],q[3];
x q[12];
cx q[19],q[13];
s q[6];
cx q[17],q[19];
cx q[12],q[10];
cx q[1],q[13];
cx q[6],q[18];
cx q[9],q[19];
cx q[13],q[5];
cx q[6],q[4];
cx q[17],q[3];
cx q[16],q[5];
cx q[12],q[11];
cx q[13],q[2];
cx q[5],q[10];
cx q[2],q[3];
cx q[16],q[5];
cx q[5],q[15];
cx q[7],q[4];
cx q[13],q[8];
cx q[1],q[2];
cx q[12],q[5];
cx q[13],q[1];
cx q[15],q[3];
cx q[1],q[0];
s q[17];
cx q[11],q[13];
cx q[3],q[18];
cx q[1],q[11];
cx q[0],q[11];
cx q[19],q[12];
cx q[19],q[4];
cx q[6],q[12];
cx q[18],q[17];
cx q[19],q[12];
cx q[18],q[11];
cx q[11],q[10];
cx q[13],q[16];
cx q[1],q[9];
cx q[0],q[11];
x q[9];
cx q[2],q[3];
t q[0];
cx q[13],q[1];
cx q[7],q[10];
cx q[0],q[7];
cx q[2],q[5];
cx q[4],q[1];
cx q[2],q[7];
cx q[4],q[12];
cx q[2],q[18];
cx q[2],q[18];
cx q[19],q[9];
cx q[13],q[12];
cx q[7],q[18];
cx q[16],q[9];
cx q[4],q[16];
cx q[17],q[7];
cx q[4],q[2];
cx q[3],q[9];
cx q[2],q[0];
cx q[9],q[4];
cx q[8],q[4];
cx q[12],q[19];
cx q[3],q[18];
cx q[1],q[15];
cx q[14],q[2];
cx q[3],q[18];
cx q[3],q[1];
cx q[1],q[3];
cx q[0],q[19];
cx q[10],q[16];
h q[4];
cx q[17],q[1];
cx q[14],q[16];
cx q[12],q[17];
cx q[1],q[12];
cx q[5],q[2];
cx q[1],q[2];
cx q[12],q[3];
cx q[0],q[1];
cx q[13],q[12];
cx q[8],q[18];
cx q[12],q[9];
cx q[2],q[12];
cx q[10],q[1];
cx q[18],q[19];
cx q[13],q[18];
cx q[6],q[0];
cx q[17],q[12];
cx q[9],q[3];
cx q[9],q[1];
s q[12];
cx q[13],q[19];
cx q[3],q[14];
cx q[17],q[1];
cx q[9],q[10];
h q[6];
cx q[12],q[14];
cx q[0],q[2];
cx q[5],q[4];
s q[16];
cx q[16],q[5];
cx q[5],q[12];
cx q[10],q[11];
cx q[10],q[17];
h q[12];
cx q[14],q[15];
s q[3];
cx q[10],q[5];
cx q[8],q[17];
cx q[13],q[0];
cx q[2],q[4];
cx q[9],q[8];
cx q[3],q[8];
cx q[8],q[11];
cx q[12],q[1];
cx q[0],q[8];
cx q[0],q[11];
cx q[14],q[5];
cx q[4],q[15];
cx q[4],q[13];
cx q[7],q[0];
cx q[4],q[18];
cx q[17],q[14];
cx q[8],q[2];
cx q[10],q[18];
cx q[10],q[6];
cx q[8],q[11];
cx q[0],q[16];
cx q[2],q[13];
cx q[0],q[10];
cx q[10],q[4];
cx q[10],q[13];
cx q[11],q[4];
cx q[3],q[11];
cx q[8],q[10];
cx q[0],q[15];
cx q[19],q[18];
cx q[2],q[5];
cx q[12],q[1];
cx q[12],q[16];
cx q[12],q[7];
cx q[12],q[1];
cx q[10],q[3];
x q[13];
cx q[9],q[13];
cx q[14],q[9];
cx q[19],q[8];
cx q[17],q[14];
cx q[3],q[2];
cx q[10],q[3];
x q[15];
cx q[3],q[7];
cx q[12],q[16];
cx q[11],q[12];
cx q[13],q[0];
cx q[18],q[3];
cx q[6],q[16];
cx q[15],q[19];
cx q[9],q[0];
cx q[5],q[6];
cx q[6],q[2];
cx q[8],q[1];
cx q[2],q[14];
cx q[8],q[3];
cx q[16],q[11];
cx q[5],q[19];
cx q[8],q[17];
cx q[11],q[2];
cx q[11],q[15];
cx q[9],q[18];
cx q[19],q[0];
cx q[10],q[19];
cx q[18],q[8];
cx q[14],q[12];
cx q[8],q[11];
cx q[16],q[15];
cx q[17],q[13];
cx q[19],q[8];
cx q[5],q[8];
cx q[19],q[13];
cx q[14],q[10];
cx q[2],q[15];
cx q[15],q[14];
cx q[5],q[8];
cx q[2],q[0];
cx q[4],q[9];
cx q[17],q[11];
h q[1];
cx q[1],q[4];
cx q[15],q[16];
cx q[5],q[1];
cx q[9],q[12];
cx q[3],q[10];
cx q[2],q[15];
cx q[1],q[9];
cx q[2],q[8];
cx q[10],q[6];
cx q[13],q[9];
cx q[17],q[11];
cx q[13],q[0];
cx q[3],q[14];
cx q[0],q[3];
cx q[16],q[11];
cx q[5],q[18];
cx q[1],q[10];
cx q[12],q[11];
cx q[10],q[4];
cx q[6],q[14];
cx q[16],q[11];
cx q[10],q[17];
cx q[3],q[6];
x q[2];
cx q[13],q[16];
cx q[10],q[4];
cx q[1],q[6];
cx q[3],q[19];
cx q[13],q[17];
t q[16];
cx q[12],q[1];
cx q[12],q[9];
cx q[19],q[5];
cx q[9],q[2];
cx q[7],q[3];
cx q[15],q[12];
cx q[1],q[15];
cx q[13],q[15];
cx q[19],q[15];
cx q[11],q[16];
cx q[4],q[16];
cx q[2],q[15];
cx q[2],q[7];
cx q[10],q[9];
cx q[10],q[15];
cx q[13],q[18];
cx q[6],q[11];
cx q[17],q[12];
cx q[18],q[1];
t q[7];
cx q[7],q[0];
cx q[4],q[12];
cx q[8],q[6];
cx q[15],q[0];
cx q[13],q[0];
cx q[2],q[5];
cx q[17],q[16];
cx q[6],q[17];
cx q[2],q[10];
cx q[0],q[7];
cx q[7],q[11];
cx q[15],q[1];
cx q[16],q[14];
cx q[7],q[19];
cx q[8],q[6];
cx q[10],q[17];
cx q[15],q[18];
cx q[16],q[0];
cx q[13],q[4];
cx q[2],q[9];
cx q[9],q[6];
cx q[6],q[13];
cx q[11],q[14];
cx q[7],q[11];
cx q[13],q[17];
cx q[18],q[4];
cx q[17],q[1];
cx q[3],q[2];
cx q[7],q[19];
cx q[12],q[7];
cx q[15],q[0];
x q[18];
cx q[3],q[0];
cx q[6],q[17];
cx q[18],q[6];
cx q[19],q[14];
cx q[19],q[7];
cx q[16],q[4];
cx q[1],q[5];
cx q[16],q[19];
cx q[17],q[10];
cx q[10],q[6];
cx q[2],q[15];
cx q[9],q[18];
cx q[0],q[8];
cx q[6],q[1];
cx q[11],q[8];
cx q[5],q[12];
cx q[11],q[13];
cx q[10],q[7];
cx q[12],q[1];
cx q[9],q[7];
cx q[6],q[3];
cx q[3],q[17];
cx q[10],q[11];
cx q[7],q[11];
cx q[1],q[16];
cx q[11],q[1];
cx q[0],q[18];
cx q[10],q[18];
cx q[17],q[18];
cx q[10],q[18];
cx q[5],q[2];
cx q[6],q[11];
cx q[19],q[9];
cx q[16],q[3];
cx q[3],q[4];
cx q[10],q[9];
cx q[11],q[10];
cx q[13],q[10];
cx q[17],q[10];
cx q[6],q[19];
cx q[10],q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[14],q[3];
cx q[19],q[10];
cx q[6],q[9];
cx q[16],q[7];
cx q[19],q[6];
cx q[6],q[5];
cx q[1],q[9];
cx q[6],q[13];
cx q[1],q[0];
cx q[19],q[5];
cx q[2],q[11];
cx q[0],q[19];
cx q[4],q[7];
cx q[15],q[3];
cx q[12],q[8];
cx q[18],q[10];
cx q[15],q[14];
cx q[6],q[4];
cx q[12],q[17];
cx q[5],q[6];
cx q[11],q[1];
cx q[7],q[9];
cx q[14],q[12];
cx q[11],q[0];
cx q[12],q[9];
cx q[7],q[3];
cx q[4],q[0];
cx q[4],q[9];
cx q[10],q[15];
cx q[18],q[13];
cx q[0],q[17];
cx q[17],q[11];
cx q[18],q[11];
cx q[8],q[4];
cx q[0],q[19];
cx q[19],q[14];
cx q[12],q[7];
cx q[11],q[14];
cx q[17],q[15];
cx q[18],q[7];
h q[10];
cx q[19],q[1];
cx q[17],q[3];
cx q[7],q[11];
cx q[3],q[2];
cx q[7],q[0];
cx q[6],q[2];
cx q[4],q[16];
cx q[15],q[0];
cx q[13],q[0];
cx q[19],q[18];
cx q[11],q[5];
cx q[14],q[1];
cx q[9],q[8];
cx q[10],q[17];
cx q[5],q[0];
cx q[2],q[8];
cx q[3],q[16];
cx q[19],q[5];
cx q[11],q[8];
cx q[8],q[10];
cx q[12],q[6];
cx q[12],q[0];
cx q[9],q[17];
cx q[8],q[6];
cx q[0],q[18];
cx q[10],q[14];
cx q[15],q[3];
cx q[3],q[19];
cx q[2],q[3];
cx q[2],q[4];
cx q[16],q[10];
cx q[1],q[7];
cx q[19],q[4];
cx q[4],q[15];
h q[0];
cx q[13],q[17];
cx q[14],q[2];
cx q[17],q[4];
cx q[19],q[17];
cx q[16],q[3];
cx q[0],q[7];
cx q[7],q[16];
cx q[10],q[16];
cx q[6],q[3];
cx q[9],q[18];
cx q[8],q[5];
cx q[17],q[11];
cx q[3],q[5];
cx q[15],q[2];
cx q[0],q[5];
cx q[3],q[19];
cx q[8],q[14];
cx q[11],q[12];
cx q[13],q[5];
cx q[19],q[11];
cx q[12],q[3];
cx q[9],q[15];
cx q[18],q[12];
h q[12];
cx q[19],q[11];
cx q[15],q[17];
cx q[5],q[15];
cx q[12],q[5];
cx q[10],q[13];
cx q[9],q[14];
cx q[10],q[11];
cx q[7],q[1];
cx q[16],q[5];
cx q[3],q[7];
cx q[7],q[11];
cx q[12],q[15];
cx q[7],q[15];
cx q[4],q[1];
cx q[2],q[13];
cx q[13],q[7];
cx q[2],q[4];
cx q[16],q[11];
cx q[17],q[16];
cx q[16],q[15];
cx q[13],q[12];
cx q[0],q[12];
cx q[19],q[6];
cx q[7],q[2];
cx q[3],q[2];
cx q[7],q[18];
cx q[10],q[18];
cx q[12],q[14];
cx q[1],q[9];
cx q[19],q[13];
cx q[8],q[3];
cx q[9],q[10];
cx q[7],q[13];
cx q[1],q[15];
cx q[19],q[10];
cx q[0],q[5];
cx q[2],q[12];
cx q[2],q[13];
cx q[2],q[7];
cx q[15],q[2];
cx q[14],q[18];
cx q[7],q[4];
cx q[8],q[12];
cx q[0],q[3];
cx q[12],q[19];
cx q[11],q[3];
cx q[6],q[15];
cx q[19],q[7];
cx q[8],q[3];
cx q[5],q[18];
cx q[15],q[11];
cx q[19],q[2];
cx q[8],q[5];
cx q[6],q[5];
cx q[1],q[10];
s q[4];
cx q[19],q[8];
cx q[7],q[14];
t q[11];
cx q[10],q[15];
cx q[14],q[2];
cx q[12],q[17];
cx q[17],q[9];
cx q[16],q[10];
cx q[2],q[9];
h q[16];
cx q[6],q[9];
h q[4];
cx q[18],q[0];
cx q[12],q[13];
cx q[17],q[4];
x q[2];
h q[18];
cx q[9],q[0];
cx q[14],q[19];
cx q[3],q[11];
h q[15];
t q[8];
cx q[19],q[14];
cx q[7],q[3];
cx q[13],q[18];
h q[3];
cx q[1],q[17];
cx q[16],q[6];
cx q[12],q[6];
cx q[17],q[5];
cx q[8],q[16];
cx q[7],q[18];
cx q[0],q[14];
cx q[13],q[18];
cx q[17],q[12];
cx q[4],q[17];
cx q[10],q[7];
cx q[11],q[9];
cx q[7],q[16];
cx q[1],q[7];
cx q[15],q[13];
cx q[10],q[15];
cx q[8],q[0];
cx q[19],q[12];
cx q[8],q[19];
cx q[15],q[16];
cx q[18],q[17];
cx q[12],q[18];
cx q[2],q[3];
cx q[16],q[0];
cx q[8],q[4];
cx q[16],q[11];
cx q[8],q[3];
cx q[0],q[12];
cx q[6],q[8];
cx q[1],q[12];
cx q[9],q[10];
cx q[13],q[1];
cx q[12],q[11];
cx q[19],q[13];
cx q[14],q[6];
cx q[6],q[5];
cx q[3],q[0];
cx q[12],q[18];
cx q[14],q[9];
cx q[7],q[11];
t q[0];
cx q[5],q[16];
cx q[8],q[18];
cx q[2],q[6];
cx q[19],q[6];
cx q[8],q[15];
cx q[15],q[7];
cx q[5],q[13];
cx q[14],q[16];
cx q[3],q[17];
x q[0];
cx q[8],q[15];
cx q[16],q[10];
cx q[13],q[16];
cx q[15],q[7];
cx q[7],q[12];
cx q[8],q[9];
cx q[18],q[1];
cx q[2],q[16];
cx q[12],q[1];
cx q[17],q[13];
cx q[8],q[4];
cx q[3],q[19];
cx q[18],q[1];
cx q[14],q[13];
x q[1];
x q[18];
cx q[16],q[8];
cx q[11],q[10];
cx q[10],q[12];
cx q[8],q[17];
cx q[0],q[11];
cx q[19],q[6];
cx q[17],q[14];
cx q[17],q[9];
cx q[7],q[17];
cx q[10],q[17];
cx q[16],q[14];
cx q[11],q[13];
cx q[17],q[13];
cx q[4],q[18];
cx q[18],q[1];
cx q[0],q[8];
cx q[3],q[8];
cx q[0],q[14];
cx q[10],q[3];
cx q[4],q[2];
cx q[9],q[5];
cx q[5],q[19];
cx q[17],q[16];
cx q[4],q[15];
cx q[16],q[12];
cx q[4],q[3];
cx q[9],q[7];
cx q[10],q[6];
cx q[7],q[2];
cx q[9],q[18];
cx q[19],q[5];
cx q[2],q[12];
cx q[17],q[1];
cx q[2],q[14];
cx q[6],q[13];
cx q[17],q[9];
cx q[15],q[12];
t q[11];
cx q[15],q[8];
cx q[11],q[12];
cx q[19],q[3];
cx q[6],q[19];
cx q[13],q[19];
cx q[2],q[14];
cx q[0],q[12];
cx q[2],q[9];
cx q[13],q[14];
cx q[2],q[12];
cx q[13],q[6];
cx q[11],q[0];
cx q[13],q[5];
cx q[12],q[5];
cx q[1],q[2];
cx q[3],q[8];
cx q[7],q[9];
cx q[16],q[15];
cx q[19],q[5];
cx q[7],q[18];
cx q[4],q[13];
cx q[5],q[18];
cx q[10],q[6];
cx q[2],q[13];
cx q[14],q[11];
cx q[10],q[16];
cx q[10],q[0];
cx q[11],q[17];
cx q[7],q[19];
cx q[14],q[6];
cx q[4],q[0];
cx q[8],q[17];
cx q[3],q[16];
cx q[9],q[6];
cx q[6],q[3];
cx q[3],q[13];
cx q[2],q[10];
cx q[8],q[12];
cx q[15],q[9];
x q[11];
cx q[16],q[10];
cx q[14],q[9];
cx q[3],q[11];
cx q[14],q[0];
cx q[13],q[17];
cx q[16],q[3];
cx q[6],q[12];
cx q[16],q[3];
cx q[14],q[10];
cx q[17],q[3];
cx q[7],q[19];
cx q[18],q[11];
s q[14];
cx q[13],q[14];
x q[18];
cx q[3],q[0];
cx q[6],q[17];
cx q[5],q[2];
cx q[15],q[1];
cx q[6],q[13];
s q[8];
cx q[14],q[8];
cx q[15],q[14];
cx q[9],q[7];
cx q[11],q[3];
cx q[10],q[11];
cx q[5],q[0];
cx q[10],q[5];
cx q[5],q[3];
cx q[5],q[6];
cx q[16],q[9];
cx q[14],q[17];
cx q[4],q[15];
cx q[19],q[12];
cx q[1],q[17];
cx q[14],q[13];
cx q[6],q[11];
cx q[10],q[4];
cx q[5],q[12];
x q[0];
cx q[15],q[6];
cx q[15],q[1];
cx q[4],q[1];
cx q[17],q[4];
cx q[0],q[3];
cx q[5],q[2];
cx q[9],q[19];
cx q[15],q[8];
cx q[9],q[8];
cx q[6],q[7];
cx q[5],q[14];
cx q[13],q[0];
cx q[5],q[3];
cx q[3],q[17];
cx q[4],q[0];
cx q[8],q[11];
x q[15];
cx q[18],q[0];
cx q[5],q[9];
cx q[6],q[2];
cx q[4],q[9];
cx q[19],q[16];
cx q[14],q[6];
cx q[1],q[5];
cx q[9],q[16];
cx q[19],q[13];
cx q[2],q[16];
cx q[13],q[6];
cx q[4],q[12];
cx q[7],q[8];
cx q[3],q[17];
cx q[14],q[4];
cx q[16],q[3];
cx q[4],q[3];
cx q[19],q[3];
cx q[0],q[17];
cx q[7],q[11];
cx q[2],q[7];
x q[8];
x q[19];
cx q[16],q[0];
cx q[14],q[13];
cx q[3],q[9];
cx q[14],q[16];
cx q[0],q[16];
cx q[16],q[12];
cx q[3],q[7];
cx q[4],q[3];
cx q[8],q[17];
cx q[15],q[8];
cx q[11],q[9];
cx q[19],q[2];
cx q[9],q[15];
cx q[18],q[10];
cx q[16],q[0];
cx q[7],q[13];
cx q[15],q[5];
cx q[13],q[14];
cx q[7],q[0];
cx q[7],q[6];
cx q[5],q[18];
cx q[3],q[1];
cx q[3],q[0];
cx q[9],q[18];
cx q[11],q[14];
cx q[3],q[16];
cx q[8],q[1];
cx q[0],q[12];
cx q[1],q[16];
cx q[1],q[8];
cx q[17],q[15];
cx q[17],q[0];
cx q[17],q[12];
cx q[14],q[2];
cx q[19],q[8];
cx q[9],q[5];
cx q[16],q[5];
cx q[0],q[13];
cx q[0],q[4];
cx q[10],q[5];
cx q[11],q[8];
cx q[19],q[7];
cx q[8],q[15];
cx q[12],q[15];
cx q[1],q[6];
cx q[3],q[18];
cx q[3],q[16];
cx q[8],q[16];
cx q[1],q[15];
cx q[15],q[2];
cx q[8],q[1];
cx q[3],q[6];
s q[19];
cx q[15],q[4];
cx q[11],q[6];
cx q[12],q[14];
cx q[8],q[16];
cx q[3],q[13];
cx q[8],q[11];
cx q[16],q[15];
cx q[1],q[0];
cx q[7],q[8];
cx q[8],q[15];
cx q[5],q[18];
cx q[14],q[4];
cx q[13],q[12];
cx q[17],q[14];
cx q[15],q[8];
t q[16];
cx q[2],q[7];
cx q[10],q[16];

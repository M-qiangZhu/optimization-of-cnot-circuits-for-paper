// Initial wiring: [8, 9, 4, 11, 2, 0, 3, 10, 7, 13, 6, 14, 5, 12, 15, 1]
// Resulting wiring: [8, 9, 4, 11, 2, 0, 3, 10, 7, 13, 6, 14, 5, 12, 15, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5], q[2];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[6];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[2];
cx q[7], q[6];
cx q[9], q[8];
cx q[8], q[7];
cx q[9], q[8];
cx q[10], q[9];
cx q[11], q[10];
cx q[10], q[9];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[4];
cx q[14], q[9];
cx q[15], q[14];
cx q[14], q[9];
cx q[15], q[14];
cx q[13], q[14];
cx q[14], q[15];
cx q[14], q[13];
cx q[11], q[12];
cx q[12], q[11];
cx q[9], q[14];
cx q[14], q[13];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[13];
cx q[10], q[9];
cx q[6], q[9];
cx q[5], q[6];
cx q[6], q[9];
cx q[4], q[11];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[14];
cx q[9], q[10];
cx q[14], q[9];
cx q[3], q[4];
cx q[4], q[11];
cx q[11], q[12];
cx q[11], q[4];
cx q[12], q[11];
cx q[2], q[5];
cx q[5], q[10];
cx q[10], q[13];
cx q[13], q[14];
cx q[14], q[15];
cx q[5], q[6];
cx q[5], q[4];
cx q[15], q[14];
cx q[0], q[1];

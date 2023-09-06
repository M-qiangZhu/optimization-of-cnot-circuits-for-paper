// Initial wiring: [8, 17, 18, 3, 2, 19, 5, 7, 14, 10, 13, 1, 0, 4, 12, 15, 16, 11, 6, 9]
// Resulting wiring: [8, 17, 18, 3, 2, 19, 5, 7, 14, 10, 13, 1, 0, 4, 12, 15, 16, 11, 6, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[3], q[2];
cx q[6], q[4];
cx q[6], q[3];
cx q[9], q[8];
cx q[9], q[0];
cx q[13], q[6];
cx q[13], q[7];
cx q[6], q[4];
cx q[14], q[5];
cx q[16], q[13];
cx q[13], q[12];
cx q[12], q[11];
cx q[11], q[10];
cx q[13], q[12];
cx q[16], q[13];
cx q[16], q[17];
cx q[14], q[15];
cx q[13], q[16];
cx q[10], q[19];
cx q[7], q[12];
cx q[6], q[12];
cx q[6], q[7];

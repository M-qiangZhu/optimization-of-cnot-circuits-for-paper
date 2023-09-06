// Initial wiring: [5, 14, 3, 19, 9, 13, 16, 0, 2, 11, 8, 4, 1, 15, 7, 17, 12, 6, 10, 18]
// Resulting wiring: [5, 14, 3, 19, 9, 13, 16, 0, 2, 11, 8, 4, 1, 15, 7, 17, 12, 6, 10, 18]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[2], q[1];
cx q[4], q[3];
cx q[7], q[6];
cx q[10], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[11], q[10];
cx q[12], q[6];
cx q[12], q[7];
cx q[6], q[3];
cx q[13], q[6];
cx q[6], q[3];
cx q[16], q[13];
cx q[17], q[16];
cx q[16], q[13];
cx q[17], q[16];
cx q[18], q[17];
cx q[17], q[16];
cx q[16], q[13];
cx q[19], q[18];
cx q[18], q[12];
cx q[12], q[6];
cx q[18], q[12];
cx q[19], q[18];
cx q[5], q[14];
cx q[1], q[7];

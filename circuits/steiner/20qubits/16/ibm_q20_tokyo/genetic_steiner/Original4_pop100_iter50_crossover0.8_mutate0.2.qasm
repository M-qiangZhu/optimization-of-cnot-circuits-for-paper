// Initial wiring: [7, 4, 0, 1, 2, 13, 12, 10, 15, 19, 14, 16, 8, 11, 9, 17, 5, 18, 3, 6]
// Resulting wiring: [7, 4, 0, 1, 2, 13, 12, 10, 15, 19, 14, 16, 8, 11, 9, 17, 5, 18, 3, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[3], q[2];
cx q[5], q[4];
cx q[8], q[7];
cx q[14], q[13];
cx q[13], q[12];
cx q[15], q[13];
cx q[13], q[12];
cx q[13], q[6];
cx q[16], q[13];
cx q[13], q[6];
cx q[6], q[3];
cx q[16], q[14];
cx q[3], q[2];
cx q[17], q[16];
cx q[18], q[12];
cx q[13], q[14];
cx q[11], q[12];
cx q[9], q[10];
cx q[8], q[9];
cx q[7], q[13];
cx q[13], q[14];
cx q[5], q[14];

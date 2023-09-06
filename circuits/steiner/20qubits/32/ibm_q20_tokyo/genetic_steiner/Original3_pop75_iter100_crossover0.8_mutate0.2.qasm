// Initial wiring: [10, 1, 2, 4, 9, 0, 7, 6, 14, 3, 18, 12, 11, 13, 15, 19, 16, 17, 5, 8]
// Resulting wiring: [10, 1, 2, 4, 9, 0, 7, 6, 14, 3, 18, 12, 11, 13, 15, 19, 16, 17, 5, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[8], q[7];
cx q[8], q[2];
cx q[9], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[8], q[1];
cx q[11], q[8];
cx q[11], q[9];
cx q[8], q[7];
cx q[8], q[2];
cx q[8], q[1];
cx q[13], q[12];
cx q[13], q[7];
cx q[13], q[6];
cx q[14], q[13];
cx q[13], q[7];
cx q[13], q[6];
cx q[17], q[11];
cx q[11], q[9];
cx q[11], q[8];
cx q[9], q[0];
cx q[8], q[2];
cx q[17], q[11];
cx q[18], q[17];
cx q[19], q[18];
cx q[18], q[17];
cx q[18], q[12];
cx q[19], q[10];
cx q[16], q[17];
cx q[17], q[16];
cx q[12], q[18];
cx q[9], q[11];
cx q[11], q[17];
cx q[17], q[16];
cx q[16], q[17];
cx q[8], q[11];
cx q[11], q[12];
cx q[8], q[9];
cx q[11], q[8];
cx q[7], q[12];
cx q[6], q[7];
cx q[7], q[8];
cx q[6], q[12];
cx q[8], q[9];
cx q[12], q[18];
cx q[9], q[8];
cx q[4], q[6];
cx q[4], q[5];
cx q[5], q[14];
cx q[6], q[13];
cx q[3], q[6];
cx q[6], q[13];
cx q[13], q[16];
cx q[6], q[7];
cx q[1], q[2];
cx q[2], q[3];

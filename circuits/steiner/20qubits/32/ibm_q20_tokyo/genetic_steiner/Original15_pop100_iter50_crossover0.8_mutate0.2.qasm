// Initial wiring: [5, 13, 19, 14, 18, 1, 6, 3, 17, 0, 16, 10, 8, 9, 4, 12, 2, 7, 15, 11]
// Resulting wiring: [5, 13, 19, 14, 18, 1, 6, 3, 17, 0, 16, 10, 8, 9, 4, 12, 2, 7, 15, 11]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[5], q[3];
cx q[5], q[4];
cx q[3], q[2];
cx q[6], q[3];
cx q[7], q[1];
cx q[7], q[2];
cx q[1], q[0];
cx q[8], q[1];
cx q[8], q[7];
cx q[1], q[0];
cx q[10], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[3];
cx q[12], q[7];
cx q[12], q[11];
cx q[7], q[1];
cx q[11], q[8];
cx q[1], q[0];
cx q[13], q[6];
cx q[14], q[13];
cx q[13], q[6];
cx q[14], q[13];
cx q[15], q[13];
cx q[13], q[7];
cx q[7], q[1];
cx q[1], q[0];
cx q[7], q[1];
cx q[13], q[7];
cx q[17], q[12];
cx q[12], q[6];
cx q[17], q[12];
cx q[18], q[17];
cx q[18], q[11];
cx q[17], q[16];
cx q[11], q[8];
cx q[18], q[17];
cx q[19], q[18];
cx q[18], q[17];
cx q[16], q[17];
cx q[14], q[16];
cx q[16], q[17];
cx q[17], q[18];
cx q[17], q[16];
cx q[13], q[16];
cx q[11], q[17];
cx q[17], q[11];
cx q[10], q[11];
cx q[11], q[18];
cx q[11], q[17];
cx q[9], q[11];
cx q[11], q[17];
cx q[17], q[11];
cx q[6], q[7];
cx q[7], q[6];
cx q[5], q[14];
cx q[4], q[6];
cx q[6], q[7];
cx q[7], q[6];
cx q[3], q[4];
cx q[4], q[3];
cx q[2], q[3];
cx q[3], q[6];
cx q[6], q[12];
cx q[2], q[7];

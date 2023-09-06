// Initial wiring: [2, 7, 6, 3, 8, 10, 13, 9, 0, 1, 4, 12, 11, 5, 14, 15]
// Resulting wiring: [2, 7, 6, 3, 8, 10, 13, 9, 0, 1, 4, 12, 11, 5, 14, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7], q[0];
cx q[8], q[7];
cx q[7], q[6];
cx q[10], q[9];
cx q[14], q[9];
cx q[9], q[6];
cx q[14], q[13];
cx q[14], q[9];
cx q[15], q[14];
cx q[14], q[9];
cx q[9], q[6];
cx q[14], q[9];
cx q[11], q[12];
cx q[12], q[13];
cx q[10], q[13];
cx q[9], q[14];
cx q[8], q[9];
cx q[9], q[14];
cx q[6], q[7];
cx q[5], q[6];
cx q[5], q[10];
cx q[6], q[7];
cx q[10], q[13];
cx q[7], q[6];
cx q[4], q[11];
cx q[3], q[4];

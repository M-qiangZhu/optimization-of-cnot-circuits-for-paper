// Initial wiring: [0 2 1 3 5 4 6 7 8]
// Resulting wiring: [0 4 1 2 5 8 7 6 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[8], q[3];
cx q[4], q[3];
cx q[4], q[3];
cx q[4], q[3];
cx q[6], q[5];
cx q[2], q[1];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[7], q[6];
cx q[7], q[6];
cx q[3], q[2];
cx q[3], q[8];
cx q[8], q[3];
cx q[4], q[5];
cx q[8], q[7];

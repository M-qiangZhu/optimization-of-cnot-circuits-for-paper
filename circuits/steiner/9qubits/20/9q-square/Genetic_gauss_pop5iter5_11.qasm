// Initial wiring: [5 1 3 2 4 6 0 8 7]
// Resulting wiring: [6 1 3 2 7 5 0 8 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[4];
cx q[4], q[5];
cx q[5], q[6];
cx q[5], q[6];
cx q[5], q[6];
cx q[5], q[4];
cx q[7], q[6];
cx q[8], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[3], q[4];
cx q[7], q[8];
cx q[3], q[8];
cx q[0], q[1];
cx q[6], q[5];
cx q[3], q[2];

// Initial wiring: [1 5 2 3 4 0 6 7 8]
// Resulting wiring: [1 5 2 8 4 0 7 6 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[1], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[7], q[8];
cx q[0], q[1];
cx q[7], q[4];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[1], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[5], q[6];
cx q[3], q[4];

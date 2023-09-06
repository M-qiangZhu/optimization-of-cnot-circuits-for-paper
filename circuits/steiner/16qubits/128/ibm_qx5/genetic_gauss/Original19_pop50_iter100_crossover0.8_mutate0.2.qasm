// Initial wiring: [11, 13, 14, 8, 15, 1, 10, 9, 6, 4, 2, 0, 3, 5, 7, 12]
// Resulting wiring: [11, 13, 14, 8, 15, 1, 10, 9, 6, 4, 2, 0, 3, 5, 7, 12]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[2], q[0];
cx q[3], q[0];
cx q[4], q[2];
cx q[4], q[0];
cx q[5], q[3];
cx q[6], q[5];
cx q[6], q[3];
cx q[7], q[5];
cx q[7], q[3];
cx q[7], q[1];
cx q[7], q[0];
cx q[8], q[7];
cx q[8], q[0];
cx q[9], q[8];
cx q[9], q[7];
cx q[9], q[3];
cx q[10], q[9];
cx q[10], q[8];
cx q[10], q[5];
cx q[11], q[10];
cx q[11], q[9];
cx q[11], q[8];
cx q[11], q[6];
cx q[11], q[3];
cx q[11], q[1];
cx q[3], q[2];
cx q[6], q[4];
cx q[12], q[0];
cx q[13], q[0];
cx q[14], q[12];
cx q[14], q[11];
cx q[15], q[14];
cx q[15], q[1];
cx q[14], q[3];
cx q[11], q[4];
cx q[15], q[5];
cx q[15], q[7];
cx q[15], q[8];
cx q[13], q[9];
cx q[12], q[10];
cx q[13], q[14];
cx q[12], q[14];
cx q[14], q[12];
cx q[10], q[13];
cx q[13], q[10];
cx q[9], q[14];
cx q[9], q[13];
cx q[8], q[15];
cx q[8], q[11];
cx q[8], q[9];
cx q[9], q[8];
cx q[7], q[8];
cx q[6], q[7];
cx q[7], q[12];
cx q[8], q[10];
cx q[5], q[13];
cx q[5], q[9];
cx q[5], q[6];
cx q[6], q[5];
cx q[4], q[15];
cx q[4], q[10];
cx q[4], q[9];
cx q[4], q[8];
cx q[4], q[7];
cx q[4], q[6];
cx q[6], q[4];
cx q[3], q[15];
cx q[3], q[11];
cx q[3], q[6];
cx q[3], q[5];
cx q[2], q[13];
cx q[2], q[11];
cx q[2], q[10];
cx q[2], q[7];
cx q[7], q[2];
cx q[1], q[11];
cx q[1], q[7];
cx q[1], q[6];
cx q[1], q[5];
cx q[5], q[1];
cx q[0], q[10];
cx q[0], q[6];
cx q[0], q[2];
cx q[0], q[1];
cx q[1], q[14];
cx q[2], q[12];

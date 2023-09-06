// Initial wiring: [11, 0, 2, 4, 15, 5, 3, 10, 1, 7, 14, 12, 8, 6, 13, 9]
// Resulting wiring: [11, 0, 2, 4, 15, 5, 3, 10, 1, 7, 14, 12, 8, 6, 13, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[3], q[2];
cx q[5], q[2];
cx q[2], q[1];
cx q[5], q[4];
cx q[1], q[0];
cx q[4], q[3];
cx q[2], q[1];
cx q[5], q[2];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[3];
cx q[5], q[2];
cx q[6], q[1];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[3];
cx q[5], q[4];
cx q[6], q[5];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[2];
cx q[6], q[1];
cx q[7], q[6];
cx q[8], q[7];
cx q[9], q[6];
cx q[9], q[8];
cx q[6], q[5];
cx q[6], q[1];
cx q[10], q[9];
cx q[9], q[6];
cx q[10], q[9];
cx q[11], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[11], q[10];
cx q[3], q[2];
cx q[12], q[11];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[13], q[12];
cx q[12], q[11];
cx q[11], q[4];
cx q[4], q[3];
cx q[13], q[10];
cx q[3], q[2];
cx q[10], q[9];
cx q[2], q[1];
cx q[9], q[8];
cx q[9], q[6];
cx q[1], q[0];
cx q[3], q[2];
cx q[11], q[4];
cx q[10], q[9];
cx q[12], q[11];
cx q[13], q[10];
cx q[14], q[9];
cx q[9], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[14], q[13];
cx q[4], q[3];
cx q[15], q[14];
cx q[14], q[13];
cx q[13], q[10];
cx q[10], q[5];
cx q[5], q[4];
cx q[5], q[2];
cx q[4], q[3];
cx q[2], q[1];
cx q[5], q[4];
cx q[15], q[14];
cx q[14], q[15];
cx q[15], q[14];
cx q[12], q[13];
cx q[13], q[12];
cx q[11], q[12];
cx q[10], q[13];
cx q[9], q[10];
cx q[10], q[11];
cx q[11], q[12];
cx q[11], q[10];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[11], q[12];
cx q[7], q[8];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[11], q[12];
cx q[6], q[7];
cx q[6], q[9];
cx q[7], q[8];
cx q[7], q[6];
cx q[5], q[6];
cx q[6], q[9];
cx q[9], q[14];
cx q[6], q[5];
cx q[14], q[9];
cx q[4], q[5];
cx q[5], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[15];
cx q[6], q[9];
cx q[6], q[5];
cx q[7], q[6];
cx q[15], q[8];
cx q[2], q[5];
cx q[5], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[9];
cx q[6], q[5];
cx q[7], q[6];
cx q[1], q[2];
cx q[2], q[5];
cx q[5], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[8], q[15];
cx q[10], q[5];
cx q[8], q[9];
cx q[0], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[11];
cx q[5], q[10];
cx q[10], q[13];
cx q[11], q[12];

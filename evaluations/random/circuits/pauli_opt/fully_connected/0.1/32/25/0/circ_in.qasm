OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[8],q[24];
rx(7*pi/4) q[8];
cx q[8],q[24];
rx(5*pi/4) q[5];
rz(pi/4) q[18];
rz(3*pi/2) q[2];
cx q[24],q[17];
rz(7*pi/4) q[17];
cx q[24],q[17];
cx q[18],q[0];
rz(5*pi/4) q[0];
cx q[18],q[0];
cx q[10],q[18];
rx(3*pi/4) q[10];
cx q[10],q[18];
cx q[9],q[12];
rx(pi/4) q[9];
cx q[9],q[12];
cx q[2],q[19];
rx(pi/2) q[2];
cx q[2],q[19];
cx q[16],q[21];
rx(5*pi/4) q[16];
cx q[16],q[21];
rx(pi/2) q[6];
cx q[8],q[20];
rx(5*pi/4) q[8];
cx q[8],q[20];
rz(pi/2) q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
rx(pi) q[9];
rz(pi/2) q[6];
rx(5*pi/4) q[16];
cx q[22],q[14];
rz(5*pi/4) q[14];
cx q[22],q[14];
cx q[7],q[20];
rx(3*pi/4) q[7];
cx q[7],q[20];
cx q[4],q[11];
rx(3*pi/2) q[4];
cx q[4],q[11];
cx q[4],q[6];
rx(5*pi/4) q[4];
cx q[4],q[6];
cx q[22],q[0];
rz(3*pi/2) q[0];
cx q[22],q[0];
rz(5*pi/4) q[3];
cx q[2],q[12];
rx(5*pi/4) q[2];
cx q[2],q[12];
cx q[24],q[11];
rz(3*pi/4) q[11];
cx q[24],q[11];
rz(pi/4) q[16];
rx(pi/4) q[20];
rz(5*pi/4) q[18];
cx q[12],q[2];
rz(pi/4) q[2];
cx q[12],q[2];
rx(3*pi/2) q[5];
cx q[13],q[15];
rx(5*pi/4) q[13];
cx q[13],q[15];
rz(3*pi/4) q[17];
cx q[8],q[24];
rx(7*pi/4) q[8];
cx q[8],q[24];
rx(5*pi/4) q[5];
rz(pi/4) q[18];
rz(3*pi/2) q[2];
cx q[24],q[17];
rz(7*pi/4) q[17];
cx q[24],q[17];
cx q[18],q[0];
rz(5*pi/4) q[0];
cx q[18],q[0];
cx q[10],q[18];
rx(3*pi/4) q[10];
cx q[10],q[18];
cx q[9],q[12];
rx(pi/4) q[9];
cx q[9],q[12];
cx q[2],q[19];
rx(pi/2) q[2];
cx q[2],q[19];
cx q[16],q[21];
rx(5*pi/4) q[16];
cx q[16],q[21];
rx(pi/2) q[6];
cx q[8],q[20];
rx(5*pi/4) q[8];
cx q[8],q[20];
rz(pi/2) q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
rx(pi) q[9];
rz(pi/2) q[6];
rx(5*pi/4) q[16];
cx q[22],q[14];
rz(5*pi/4) q[14];
cx q[22],q[14];
cx q[7],q[20];
rx(3*pi/4) q[7];
cx q[7],q[20];
cx q[4],q[11];
rx(3*pi/2) q[4];
cx q[4],q[11];
cx q[4],q[6];
rx(5*pi/4) q[4];
cx q[4],q[6];
cx q[22],q[0];
rz(3*pi/2) q[0];
cx q[22],q[0];
rz(5*pi/4) q[3];
cx q[2],q[12];
rx(5*pi/4) q[2];
cx q[2],q[12];
cx q[24],q[11];
rz(3*pi/4) q[11];
cx q[24],q[11];
rz(pi/4) q[16];
rx(pi/4) q[20];
rz(5*pi/4) q[18];
cx q[12],q[2];
rz(pi/4) q[2];
cx q[12],q[2];
rx(3*pi/2) q[5];
cx q[13],q[15];
rx(5*pi/4) q[13];
cx q[13],q[15];
rz(3*pi/4) q[17];
cx q[8],q[24];
rx(7*pi/4) q[8];
cx q[8],q[24];
rx(5*pi/4) q[5];
rz(pi/4) q[18];
rz(3*pi/2) q[2];
cx q[24],q[17];
rz(7*pi/4) q[17];
cx q[24],q[17];
cx q[18],q[0];
rz(5*pi/4) q[0];
cx q[18],q[0];
cx q[10],q[18];
rx(3*pi/4) q[10];
cx q[10],q[18];
cx q[9],q[12];
rx(pi/4) q[9];
cx q[9],q[12];
cx q[2],q[19];
rx(pi/2) q[2];
cx q[2],q[19];
cx q[16],q[21];
rx(5*pi/4) q[16];
cx q[16],q[21];
rx(pi/2) q[6];
cx q[8],q[20];
rx(5*pi/4) q[8];
cx q[8],q[20];
rz(pi/2) q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
rx(pi) q[9];
rz(pi/2) q[6];
rx(5*pi/4) q[16];
cx q[22],q[14];
rz(5*pi/4) q[14];
cx q[22],q[14];
cx q[7],q[20];
rx(3*pi/4) q[7];
cx q[7],q[20];
cx q[4],q[11];
rx(3*pi/2) q[4];
cx q[4],q[11];
cx q[4],q[6];
rx(5*pi/4) q[4];
cx q[4],q[6];
cx q[22],q[0];
rz(3*pi/2) q[0];
cx q[22],q[0];
rz(5*pi/4) q[3];
cx q[2],q[12];
rx(5*pi/4) q[2];
cx q[2],q[12];
cx q[24],q[11];
rz(3*pi/4) q[11];
cx q[24],q[11];
rz(pi/4) q[16];
rx(pi/4) q[20];
rz(5*pi/4) q[18];
cx q[12],q[2];
rz(pi/4) q[2];
cx q[12],q[2];
rx(3*pi/2) q[5];
cx q[13],q[15];
rx(5*pi/4) q[13];
cx q[13],q[15];
rz(3*pi/4) q[17];
cx q[8],q[24];
rx(7*pi/4) q[8];
cx q[8],q[24];
rx(5*pi/4) q[5];
rz(pi/4) q[18];
rz(3*pi/2) q[2];
cx q[24],q[17];
rz(7*pi/4) q[17];
cx q[24],q[17];
cx q[18],q[0];
rz(5*pi/4) q[0];
cx q[18],q[0];
cx q[10],q[18];
rx(3*pi/4) q[10];
cx q[10],q[18];
cx q[9],q[12];
rx(pi/4) q[9];
cx q[9],q[12];
cx q[2],q[19];
rx(pi/2) q[2];
cx q[2],q[19];
cx q[16],q[21];
rx(5*pi/4) q[16];
cx q[16],q[21];
rx(pi/2) q[6];
cx q[8],q[20];
rx(5*pi/4) q[8];
cx q[8],q[20];
rz(pi/2) q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
rx(pi) q[9];
rz(pi/2) q[6];
rx(5*pi/4) q[16];
cx q[22],q[14];
rz(5*pi/4) q[14];
cx q[22],q[14];
cx q[7],q[20];
rx(3*pi/4) q[7];
cx q[7],q[20];
cx q[4],q[11];
rx(3*pi/2) q[4];
cx q[4],q[11];
cx q[4],q[6];
rx(5*pi/4) q[4];
cx q[4],q[6];
cx q[22],q[0];
rz(3*pi/2) q[0];
cx q[22],q[0];
rz(5*pi/4) q[3];
cx q[2],q[12];
rx(5*pi/4) q[2];
cx q[2],q[12];
cx q[24],q[11];
rz(3*pi/4) q[11];
cx q[24],q[11];
rz(pi/4) q[16];
rx(pi/4) q[20];
rz(5*pi/4) q[18];
cx q[12],q[2];
rz(pi/4) q[2];
cx q[12],q[2];
rx(3*pi/2) q[5];
cx q[13],q[15];
rx(5*pi/4) q[13];
cx q[13],q[15];
rz(3*pi/4) q[17];
cx q[8],q[24];
rx(7*pi/4) q[8];
cx q[8],q[24];
rx(5*pi/4) q[5];
rz(pi/4) q[18];
rz(3*pi/2) q[2];
cx q[24],q[17];
rz(7*pi/4) q[17];
cx q[24],q[17];
cx q[18],q[0];
rz(5*pi/4) q[0];
cx q[18],q[0];
cx q[10],q[18];
rx(3*pi/4) q[10];
cx q[10],q[18];
cx q[9],q[12];
rx(pi/4) q[9];
cx q[9],q[12];
cx q[2],q[19];
rx(pi/2) q[2];
cx q[2],q[19];
cx q[16],q[21];
rx(5*pi/4) q[16];
cx q[16],q[21];
rx(pi/2) q[6];
cx q[8],q[20];
rx(5*pi/4) q[8];
cx q[8],q[20];
rz(pi/2) q[7];
cx q[2],q[5];
rx(3*pi/4) q[2];
cx q[2],q[5];
rx(pi) q[9];
rz(pi/2) q[6];
rx(5*pi/4) q[16];
cx q[22],q[14];
rz(5*pi/4) q[14];
cx q[22],q[14];
cx q[7],q[20];
rx(3*pi/4) q[7];
cx q[7],q[20];
cx q[4],q[11];
rx(3*pi/2) q[4];
cx q[4],q[11];
cx q[4],q[6];
rx(5*pi/4) q[4];
cx q[4],q[6];
cx q[22],q[0];
rz(3*pi/2) q[0];
cx q[22],q[0];
rz(5*pi/4) q[3];
cx q[2],q[12];
rx(5*pi/4) q[2];
cx q[2],q[12];
cx q[24],q[11];
rz(3*pi/4) q[11];
cx q[24],q[11];
rz(pi/4) q[16];
rx(pi/4) q[20];
rz(5*pi/4) q[18];
cx q[12],q[2];
rz(pi/4) q[2];
cx q[12],q[2];
rx(3*pi/2) q[5];
cx q[13],q[15];
rx(5*pi/4) q[13];
cx q[13],q[15];
rz(3*pi/4) q[17];
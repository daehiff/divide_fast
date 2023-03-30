OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[5],q[14];
cx q[15],q[16];
cx q[8],q[9];
cx q[3],q[6];
cx q[12],q[11];
cx q[3],q[4];
cx q[15],q[14];
cx q[1],q[8];
cx q[17],q[22];
cx q[11],q[12];
cx q[8],q[11];
cx q[9],q[0];
cx q[12],q[13];
cx q[14],q[15];
cx q[3],q[6];
rx(pi) q[3];
cx q[3],q[6];
rx(pi) q[0];
rx(pi) q[10];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[13],q[12];
cx q[6],q[13];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
cx q[6],q[13];
cx q[13],q[12];
cx q[12],q[17];
cx q[15],q[14];
cx q[16],q[15];
cx q[17],q[16];
cx q[9],q[17];
rx(pi/2) q[9];
cx q[9],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
rx(3*pi/4) q[23];
cx q[16],q[15];
cx q[13],q[16];
cx q[12],q[13];
rx(7*pi/4) q[12];
cx q[12],q[13];
cx q[13],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[16],q[15];
cx q[12],q[16];
cx q[11],q[12];
cx q[8],q[11];
rx(3*pi/2) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[12],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[12],q[11];
cx q[11],q[8];
rz(7*pi/4) q[8];
cx q[11],q[8];
cx q[12],q[11];
cx q[9],q[8];
cx q[8],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[8],q[5];
cx q[9],q[8];
cx q[3],q[6];
rx(pi) q[3];
cx q[3],q[6];
rx(pi) q[0];
rx(pi) q[10];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[13],q[12];
cx q[6],q[13];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
cx q[6],q[13];
cx q[13],q[12];
cx q[12],q[17];
cx q[15],q[14];
cx q[16],q[15];
cx q[17],q[16];
cx q[9],q[17];
rx(pi/2) q[9];
cx q[9],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
rx(3*pi/4) q[23];
cx q[16],q[15];
cx q[13],q[16];
cx q[12],q[13];
rx(7*pi/4) q[12];
cx q[12],q[13];
cx q[13],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[16],q[15];
cx q[12],q[16];
cx q[11],q[12];
cx q[8],q[11];
rx(3*pi/2) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[12],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[12],q[11];
cx q[11],q[8];
rz(7*pi/4) q[8];
cx q[11],q[8];
cx q[12],q[11];
cx q[9],q[8];
cx q[8],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[8],q[5];
cx q[9],q[8];
cx q[3],q[6];
rx(pi) q[3];
cx q[3],q[6];
rx(pi) q[0];
rx(pi) q[10];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[13],q[12];
cx q[6],q[13];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
cx q[6],q[13];
cx q[13],q[12];
cx q[12],q[17];
cx q[15],q[14];
cx q[16],q[15];
cx q[17],q[16];
cx q[9],q[17];
rx(pi/2) q[9];
cx q[9],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
rx(3*pi/4) q[23];
cx q[16],q[15];
cx q[13],q[16];
cx q[12],q[13];
rx(7*pi/4) q[12];
cx q[12],q[13];
cx q[13],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[16],q[15];
cx q[12],q[16];
cx q[11],q[12];
cx q[8],q[11];
rx(3*pi/2) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[12],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[12],q[11];
cx q[11],q[8];
rz(7*pi/4) q[8];
cx q[11],q[8];
cx q[12],q[11];
cx q[9],q[8];
cx q[8],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[8],q[5];
cx q[9],q[8];
cx q[3],q[6];
rx(pi) q[3];
cx q[3],q[6];
rx(pi) q[0];
rx(pi) q[10];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[13],q[12];
cx q[6],q[13];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
cx q[6],q[13];
cx q[13],q[12];
cx q[12],q[17];
cx q[15],q[14];
cx q[16],q[15];
cx q[17],q[16];
cx q[9],q[17];
rx(pi/2) q[9];
cx q[9],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
rx(3*pi/4) q[23];
cx q[16],q[15];
cx q[13],q[16];
cx q[12],q[13];
rx(7*pi/4) q[12];
cx q[12],q[13];
cx q[13],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[16],q[15];
cx q[12],q[16];
cx q[11],q[12];
cx q[8],q[11];
rx(3*pi/2) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[12],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[12],q[11];
cx q[11],q[8];
rz(7*pi/4) q[8];
cx q[11],q[8];
cx q[12],q[11];
cx q[9],q[8];
cx q[8],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[8],q[5];
cx q[9],q[8];
cx q[3],q[6];
rx(pi) q[3];
cx q[3],q[6];
rx(pi) q[0];
rx(pi) q[10];
cx q[11],q[12];
rx(pi) q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[13],q[12];
cx q[6],q[13];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
cx q[6],q[13];
cx q[13],q[12];
cx q[12],q[17];
cx q[15],q[14];
cx q[16],q[15];
cx q[17],q[16];
cx q[9],q[17];
rx(pi/2) q[9];
cx q[9],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
rx(3*pi/4) q[23];
cx q[16],q[15];
cx q[13],q[16];
cx q[12],q[13];
rx(7*pi/4) q[12];
cx q[12],q[13];
cx q[13],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[16],q[15];
cx q[12],q[16];
cx q[11],q[12];
cx q[8],q[11];
rx(3*pi/2) q[8];
cx q[8],q[11];
cx q[11],q[12];
cx q[12],q[16];
cx q[16],q[15];
cx q[16],q[23];
cx q[12],q[11];
cx q[11],q[8];
rz(7*pi/4) q[8];
cx q[11],q[8];
cx q[12],q[11];
cx q[9],q[8];
cx q[8],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[8],q[5];
cx q[9],q[8];
cx q[8],q[11];
cx q[9],q[0];
cx q[12],q[13];
cx q[14],q[15];
cx q[3],q[4];
cx q[15],q[14];
cx q[1],q[8];
cx q[17],q[22];
cx q[11],q[12];
cx q[5],q[14];
cx q[15],q[16];
cx q[8],q[9];
cx q[3],q[6];
cx q[12],q[11];
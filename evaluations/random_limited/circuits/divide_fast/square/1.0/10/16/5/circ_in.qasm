OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[11],q[12];
cx q[12],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[1],q[6];
cx q[6],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[6],q[1];
cx q[1],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[12],q[11];
cx q[11],q[12];
rx(pi/2) q[11];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[0];
rz(7*pi/4) q[0];
cx q[7],q[0];
cx q[8],q[7];
cx q[7],q[8];
cx q[9],q[6];
cx q[6],q[9];
cx q[1],q[6];
rx(pi) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[9],q[6];
cx q[4],q[11];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[6],q[1];
cx q[1],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[6],q[1];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
cx q[9],q[6];
cx q[7],q[6];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[9],q[6];
rx(pi) q[14];
rx(pi) q[9];
cx q[13],q[10];
cx q[10],q[13];
cx q[5],q[10];
cx q[5],q[2];
cx q[2],q[5];
cx q[1],q[2];
rx(3*pi/2) q[1];
cx q[1],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[5],q[10];
cx q[10],q[13];
cx q[13],q[10];
rz(5*pi/4) q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[1],q[6];
cx q[6],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[6],q[1];
cx q[1],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[12],q[11];
cx q[11],q[12];
rx(pi/2) q[11];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[0];
rz(7*pi/4) q[0];
cx q[7],q[0];
cx q[8],q[7];
cx q[7],q[8];
cx q[9],q[6];
cx q[6],q[9];
cx q[1],q[6];
rx(pi) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[9],q[6];
cx q[4],q[11];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[6],q[1];
cx q[1],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[6],q[1];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
cx q[9],q[6];
cx q[7],q[6];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[9],q[6];
rx(pi) q[14];
rx(pi) q[9];
cx q[13],q[10];
cx q[10],q[13];
cx q[5],q[10];
cx q[5],q[2];
cx q[2],q[5];
cx q[1],q[2];
rx(3*pi/2) q[1];
cx q[1],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[5],q[10];
cx q[10],q[13];
cx q[13],q[10];
rz(5*pi/4) q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[1],q[6];
cx q[6],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[6],q[1];
cx q[1],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[12],q[11];
cx q[11],q[12];
rx(pi/2) q[11];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[0];
rz(7*pi/4) q[0];
cx q[7],q[0];
cx q[8],q[7];
cx q[7],q[8];
cx q[9],q[6];
cx q[6],q[9];
cx q[1],q[6];
rx(pi) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[9],q[6];
cx q[4],q[11];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[6],q[1];
cx q[1],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[6],q[1];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
cx q[9],q[6];
cx q[7],q[6];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[9],q[6];
rx(pi) q[14];
rx(pi) q[9];
cx q[13],q[10];
cx q[10],q[13];
cx q[5],q[10];
cx q[5],q[2];
cx q[2],q[5];
cx q[1],q[2];
rx(3*pi/2) q[1];
cx q[1],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[5],q[10];
cx q[10],q[13];
cx q[13],q[10];
rz(5*pi/4) q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[1],q[6];
cx q[6],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[6],q[1];
cx q[1],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[12],q[11];
cx q[11],q[12];
rx(pi/2) q[11];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[0];
rz(7*pi/4) q[0];
cx q[7],q[0];
cx q[8],q[7];
cx q[7],q[8];
cx q[9],q[6];
cx q[6],q[9];
cx q[1],q[6];
rx(pi) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[9],q[6];
cx q[4],q[11];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[6],q[1];
cx q[1],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[6],q[1];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
cx q[9],q[6];
cx q[7],q[6];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[9],q[6];
rx(pi) q[14];
rx(pi) q[9];
cx q[13],q[10];
cx q[10],q[13];
cx q[5],q[10];
cx q[5],q[2];
cx q[2],q[5];
cx q[1],q[2];
rx(3*pi/2) q[1];
cx q[1],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[5],q[10];
cx q[10],q[13];
cx q[13],q[10];
rz(5*pi/4) q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[1],q[6];
cx q[6],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[6],q[1];
cx q[1],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[12],q[11];
cx q[11],q[12];
rx(pi/2) q[11];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[0];
rz(7*pi/4) q[0];
cx q[7],q[0];
cx q[8],q[7];
cx q[7],q[8];
cx q[9],q[6];
cx q[6],q[9];
cx q[1],q[6];
rx(pi) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[9],q[6];
cx q[4],q[11];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[6],q[1];
cx q[1],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[6],q[1];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
cx q[9],q[6];
cx q[7],q[6];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[9],q[6];
rx(pi) q[14];
rx(pi) q[9];
cx q[13],q[10];
cx q[10],q[13];
cx q[5],q[10];
cx q[5],q[2];
cx q[2],q[5];
cx q[1],q[2];
rx(3*pi/2) q[1];
cx q[1],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[5],q[10];
cx q[10],q[13];
cx q[13],q[10];
rz(5*pi/4) q[11];

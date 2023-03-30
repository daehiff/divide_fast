OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3],q[4];
cx q[11],q[12];
cx q[14],q[9];
cx q[1],q[6];
cx q[2],q[5];
cx q[6],q[1];
cx q[5],q[4];
cx q[10],q[11];
cx q[13],q[12];
cx q[13],q[14];
cx q[2],q[1];
cx q[4],q[3];
cx q[7],q[0];
cx q[12],q[11];
cx q[11],q[4];
rz(pi/4) q[4];
cx q[11],q[4];
cx q[8],q[14];
rx(pi/2) q[8];
cx q[8],q[14];
rx(3*pi/2) q[15];
cx q[9],q[14];
cx q[9],q[8];
cx q[10],q[12];
cx q[10],q[9];
cx q[5],q[10];
rx(5*pi/4) q[5];
cx q[5],q[10];
cx q[10],q[9];
cx q[10],q[12];
cx q[9],q[8];
cx q[9],q[14];
rx(7*pi/4) q[14];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[10],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/4) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[10],q[8];
cx q[12],q[11];
cx q[7],q[9];
cx q[11],q[10];
cx q[9],q[10];
cx q[10],q[2];
rz(pi/4) q[2];
cx q[10],q[2];
cx q[9],q[10];
cx q[11],q[10];
cx q[7],q[9];
cx q[12],q[11];
rz(pi/4) q[9];
rz(3*pi/2) q[14];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[11],q[12];
cx q[4],q[11];
cx q[3],q[4];
cx q[1],q[3];
cx q[1],q[6];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[10];
cx q[11],q[10];
cx q[8],q[9];
cx q[10],q[9];
cx q[9],q[6];
rz(5*pi/4) q[6];
cx q[9],q[6];
cx q[10],q[9];
cx q[8],q[9];
cx q[11],q[10];
cx q[13],q[10];
cx q[12],q[11];
cx q[12],q[11];
cx q[11],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[11],q[4];
cx q[12],q[11];
cx q[5],q[4];
cx q[9],q[14];
cx q[6],q[5];
cx q[6],q[9];
cx q[1],q[6];
rx(3*pi/4) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[6],q[5];
cx q[9],q[14];
cx q[5],q[4];
cx q[6],q[9];
cx q[6],q[7];
cx q[4],q[11];
cx q[5],q[6];
cx q[5],q[4];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[4];
cx q[5],q[6];
cx q[4],q[11];
cx q[6],q[7];
cx q[6],q[9];
cx q[11],q[4];
rz(pi/4) q[4];
cx q[11],q[4];
cx q[8],q[14];
rx(pi/2) q[8];
cx q[8],q[14];
rx(3*pi/2) q[15];
cx q[9],q[14];
cx q[9],q[8];
cx q[10],q[12];
cx q[10],q[9];
cx q[5],q[10];
rx(5*pi/4) q[5];
cx q[5],q[10];
cx q[10],q[9];
cx q[10],q[12];
cx q[9],q[8];
cx q[9],q[14];
rx(7*pi/4) q[14];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[10],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/4) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[10],q[8];
cx q[12],q[11];
cx q[7],q[9];
cx q[11],q[10];
cx q[9],q[10];
cx q[10],q[2];
rz(pi/4) q[2];
cx q[10],q[2];
cx q[9],q[10];
cx q[11],q[10];
cx q[7],q[9];
cx q[12],q[11];
rz(pi/4) q[9];
rz(3*pi/2) q[14];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[11],q[12];
cx q[4],q[11];
cx q[3],q[4];
cx q[1],q[3];
cx q[1],q[6];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[10];
cx q[11],q[10];
cx q[8],q[9];
cx q[10],q[9];
cx q[9],q[6];
rz(5*pi/4) q[6];
cx q[9],q[6];
cx q[10],q[9];
cx q[8],q[9];
cx q[11],q[10];
cx q[13],q[10];
cx q[12],q[11];
cx q[12],q[11];
cx q[11],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[11],q[4];
cx q[12],q[11];
cx q[5],q[4];
cx q[9],q[14];
cx q[6],q[5];
cx q[6],q[9];
cx q[1],q[6];
rx(3*pi/4) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[6],q[5];
cx q[9],q[14];
cx q[5],q[4];
cx q[6],q[9];
cx q[6],q[7];
cx q[4],q[11];
cx q[5],q[6];
cx q[5],q[4];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[4];
cx q[5],q[6];
cx q[4],q[11];
cx q[6],q[7];
cx q[6],q[9];
cx q[11],q[4];
rz(pi/4) q[4];
cx q[11],q[4];
cx q[8],q[14];
rx(pi/2) q[8];
cx q[8],q[14];
rx(3*pi/2) q[15];
cx q[9],q[14];
cx q[9],q[8];
cx q[10],q[12];
cx q[10],q[9];
cx q[5],q[10];
rx(5*pi/4) q[5];
cx q[5],q[10];
cx q[10],q[9];
cx q[10],q[12];
cx q[9],q[8];
cx q[9],q[14];
rx(7*pi/4) q[14];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[10],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/4) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[10],q[8];
cx q[12],q[11];
cx q[7],q[9];
cx q[11],q[10];
cx q[9],q[10];
cx q[10],q[2];
rz(pi/4) q[2];
cx q[10],q[2];
cx q[9],q[10];
cx q[11],q[10];
cx q[7],q[9];
cx q[12],q[11];
rz(pi/4) q[9];
rz(3*pi/2) q[14];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[11],q[12];
cx q[4],q[11];
cx q[3],q[4];
cx q[1],q[3];
cx q[1],q[6];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[10];
cx q[11],q[10];
cx q[8],q[9];
cx q[10],q[9];
cx q[9],q[6];
rz(5*pi/4) q[6];
cx q[9],q[6];
cx q[10],q[9];
cx q[8],q[9];
cx q[11],q[10];
cx q[13],q[10];
cx q[12],q[11];
cx q[12],q[11];
cx q[11],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[11],q[4];
cx q[12],q[11];
cx q[5],q[4];
cx q[9],q[14];
cx q[6],q[5];
cx q[6],q[9];
cx q[1],q[6];
rx(3*pi/4) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[6],q[5];
cx q[9],q[14];
cx q[5],q[4];
cx q[6],q[9];
cx q[6],q[7];
cx q[4],q[11];
cx q[5],q[6];
cx q[5],q[4];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[4];
cx q[5],q[6];
cx q[4],q[11];
cx q[6],q[7];
cx q[6],q[9];
cx q[11],q[4];
rz(pi/4) q[4];
cx q[11],q[4];
cx q[8],q[14];
rx(pi/2) q[8];
cx q[8],q[14];
rx(3*pi/2) q[15];
cx q[9],q[14];
cx q[9],q[8];
cx q[10],q[12];
cx q[10],q[9];
cx q[5],q[10];
rx(5*pi/4) q[5];
cx q[5],q[10];
cx q[10],q[9];
cx q[10],q[12];
cx q[9],q[8];
cx q[9],q[14];
rx(7*pi/4) q[14];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[10],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/4) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[10],q[8];
cx q[12],q[11];
cx q[7],q[9];
cx q[11],q[10];
cx q[9],q[10];
cx q[10],q[2];
rz(pi/4) q[2];
cx q[10],q[2];
cx q[9],q[10];
cx q[11],q[10];
cx q[7],q[9];
cx q[12],q[11];
rz(pi/4) q[9];
rz(3*pi/2) q[14];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[11],q[12];
cx q[4],q[11];
cx q[3],q[4];
cx q[1],q[3];
cx q[1],q[6];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[10];
cx q[11],q[10];
cx q[8],q[9];
cx q[10],q[9];
cx q[9],q[6];
rz(5*pi/4) q[6];
cx q[9],q[6];
cx q[10],q[9];
cx q[8],q[9];
cx q[11],q[10];
cx q[13],q[10];
cx q[12],q[11];
cx q[12],q[11];
cx q[11],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[11],q[4];
cx q[12],q[11];
cx q[5],q[4];
cx q[9],q[14];
cx q[6],q[5];
cx q[6],q[9];
cx q[1],q[6];
rx(3*pi/4) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[6],q[5];
cx q[9],q[14];
cx q[5],q[4];
cx q[6],q[9];
cx q[6],q[7];
cx q[4],q[11];
cx q[5],q[6];
cx q[5],q[4];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[4];
cx q[5],q[6];
cx q[4],q[11];
cx q[6],q[7];
cx q[6],q[9];
cx q[11],q[4];
rz(pi/4) q[4];
cx q[11],q[4];
cx q[8],q[14];
rx(pi/2) q[8];
cx q[8],q[14];
rx(3*pi/2) q[15];
cx q[9],q[14];
cx q[9],q[8];
cx q[10],q[12];
cx q[10],q[9];
cx q[5],q[10];
rx(5*pi/4) q[5];
cx q[5],q[10];
cx q[10],q[9];
cx q[10],q[12];
cx q[9],q[8];
cx q[9],q[14];
rx(7*pi/4) q[14];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[10],q[8];
cx q[8],q[7];
cx q[7],q[6];
rz(3*pi/4) q[6];
cx q[7],q[6];
cx q[8],q[7];
cx q[10],q[8];
cx q[12],q[11];
cx q[7],q[9];
cx q[11],q[10];
cx q[9],q[10];
cx q[10],q[2];
rz(pi/4) q[2];
cx q[10],q[2];
cx q[9],q[10];
cx q[11],q[10];
cx q[7],q[9];
cx q[12],q[11];
rz(pi/4) q[9];
rz(3*pi/2) q[14];
cx q[5],q[10];
rx(pi/2) q[5];
cx q[5],q[10];
cx q[11],q[12];
cx q[4],q[11];
cx q[3],q[4];
cx q[1],q[3];
cx q[1],q[6];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[1],q[3];
cx q[3],q[4];
cx q[4],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[10];
cx q[11],q[10];
cx q[8],q[9];
cx q[10],q[9];
cx q[9],q[6];
rz(5*pi/4) q[6];
cx q[9],q[6];
cx q[10],q[9];
cx q[8],q[9];
cx q[11],q[10];
cx q[13],q[10];
cx q[12],q[11];
cx q[12],q[11];
cx q[11],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[11],q[4];
cx q[12],q[11];
cx q[5],q[4];
cx q[9],q[14];
cx q[6],q[5];
cx q[6],q[9];
cx q[1],q[6];
rx(3*pi/4) q[1];
cx q[1],q[6];
cx q[6],q[9];
cx q[6],q[5];
cx q[9],q[14];
cx q[5],q[4];
cx q[6],q[9];
cx q[6],q[7];
cx q[4],q[11];
cx q[5],q[6];
cx q[5],q[4];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[4];
cx q[5],q[6];
cx q[4],q[11];
cx q[6],q[7];
cx q[6],q[9];
cx q[13],q[14];
cx q[2],q[1];
cx q[4],q[3];
cx q[7],q[0];
cx q[12],q[11];
cx q[6],q[1];
cx q[5],q[4];
cx q[10],q[11];
cx q[13],q[12];
cx q[3],q[4];
cx q[11],q[12];
cx q[14],q[9];
cx q[1],q[6];
cx q[2],q[5];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3],q[0];
cx q[4],q[5];
rz(pi/2) q[8];
cx q[0],q[3];
cx q[5],q[4];
rx(pi/2) q[8];
cx q[3],q[0];
cx q[4],q[5];
rz(pi/2) q[8];
rz(pi/2) q[0];
cx q[1],q[5];
rz(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[0];
cx q[5],q[1];
rx(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[0];
cx q[1],q[5];
rz(pi/2) q[3];
rz(pi/2) q[4];
cx q[4],q[2];
rz(pi/2) q[3];
cx q[2],q[4];
rx(pi/2) q[3];
cx q[4],q[2];
rz(pi/2) q[3];
cx q[2],q[6];
cx q[5],q[4];
cx q[6],q[2];
cx q[4],q[5];
cx q[2],q[6];
cx q[5],q[4];
cx q[5],q[1];
cx q[2],q[4];
cx q[6],q[7];
cx q[1],q[5];
cx q[4],q[2];
cx q[7],q[6];
cx q[5],q[1];
cx q[2],q[4];
cx q[6],q[7];
rz(pi/2) q[1];
cx q[6],q[2];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx(pi/2) q[1];
cx q[2],q[6];
rx(pi/2) q[4];
rx(pi/2) q[5];
rz(pi/2) q[1];
cx q[6],q[2];
rz(pi/2) q[4];
rz(pi/2) q[5];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[5];
rz(pi/2) q[6];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[5];
rx(pi/2) q[6];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[5];
rz(pi/2) q[6];
cx q[0],q[1];
cx q[2],q[6];
rz(pi/2) q[1];
cx q[6],q[2];
rx(pi/2) q[1];
cx q[2],q[6];
rz(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[6];
cx q[0],q[1];
cx q[2],q[4];
rx(pi/2) q[6];
cx q[1],q[0];
cx q[4],q[2];
rz(pi/2) q[6];
cx q[0],q[1];
cx q[3],q[0];
cx q[1],q[5];
cx q[0],q[3];
rz(pi/2) q[1];
rz(pi/2) q[5];
cx q[3],q[0];
rx(pi/2) q[1];
rx(pi/2) q[5];
rz(pi/2) q[1];
rz(pi/2) q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[5],q[1];
cx q[4],q[2];
rz(pi/2) q[1];
cx q[2],q[4];
rx(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[1];
cx q[5],q[4];
rz(pi/2) q[1];
cx q[4],q[5];
rx(pi/2) q[1];
cx q[5],q[4];
rz(pi/2) q[1];
cx q[0],q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
rz(3*pi/2) q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[5];
cx q[5],q[1];
cx q[1],q[5];
rz(pi/2) q[1];
rz(pi/2) q[5];
rx(pi/2) q[1];
rx(pi/2) q[5];
rz(pi/2) q[1];
rz(pi/2) q[5];
cx q[0],q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/2) q[0];
cx q[1],q[5];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[5];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[5];
cx q[3],q[0];
rz(pi/2) q[1];
rz(pi/2) q[5];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[5];
cx q[5],q[1];
cx q[1],q[5];
cx q[0],q[1];
cx q[5],q[4];
cx q[1],q[0];
cx q[4],q[5];
cx q[0],q[1];
cx q[5],q[4];
cx q[3],q[0];
rz(pi/2) q[4];
cx q[0],q[3];
rx(pi/2) q[4];
cx q[3],q[0];
rz(pi/2) q[4];
cx q[2],q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cx q[1],q[5];
cx q[0],q[1];
rz(pi/2) q[5];
cx q[1],q[0];
rx(pi/2) q[5];
cx q[0],q[1];
rz(pi/2) q[5];
rz(pi/2) q[1];
rz(pi/2) q[5];
rx(pi/2) q[1];
rz(pi/2) q[5];
rz(pi/2) q[1];
rx(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cx q[4],q[5];
cx q[4],q[2];
rz(pi/2) q[5];
cx q[2],q[4];
rx(pi/2) q[5];
cx q[4],q[2];
rz(pi/2) q[5];
cx q[2],q[6];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[6];
rz(pi/2) q[4];
rx(pi/2) q[6];
cx q[2],q[4];
rz(pi/2) q[6];
rz(pi) q[2];
rz(pi/2) q[4];
cx q[6],q[7];
rz(pi/2) q[2];
rx(pi/2) q[4];
cx q[7],q[6];
rx(pi/2) q[2];
rz(pi/2) q[4];
cx q[6],q[7];
rz(pi/2) q[2];
rz(pi/2) q[6];
cx q[7],q[8];
cx q[4],q[2];
rx(pi/2) q[6];
cx q[8],q[7];
cx q[2],q[4];
rz(pi/2) q[6];
cx q[7],q[8];
cx q[4],q[2];
cx q[2],q[6];
cx q[5],q[4];
cx q[4],q[5];
rz(pi/2) q[6];
cx q[5],q[4];
rx(pi/2) q[6];
cx q[5],q[1];
rz(pi/2) q[6];
rz(pi/2) q[1];
cx q[6],q[7];
rx(pi/2) q[1];
cx q[7],q[6];
rz(pi/2) q[1];
cx q[6],q[7];
rz(pi/2) q[1];
cx q[2],q[6];
rz(pi/2) q[7];
rx(pi/2) q[1];
cx q[6],q[2];
rx(pi/2) q[7];
rz(pi/2) q[1];
cx q[2],q[6];
rz(pi/2) q[7];
cx q[1],q[0];
rz(pi/2) q[6];
cx q[0],q[3];
cx q[1],q[5];
rx(pi/2) q[6];
cx q[5],q[1];
rz(pi/2) q[6];
cx q[1],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[2],q[4];
cx q[2],q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cx q[2],q[6];
cx q[6],q[2];
cx q[2],q[6];
cx q[4],q[2];
cx q[6],q[7];
cx q[2],q[4];
rz(pi/2) q[7];
cx q[4],q[2];
rx(pi/2) q[7];
cx q[5],q[4];
rz(pi/2) q[7];
cx q[4],q[5];
cx q[7],q[8];
cx q[5],q[4];
cx q[8],q[7];
rz(pi/2) q[5];
cx q[7],q[8];
rx(pi/2) q[5];
cx q[6],q[7];
rz(pi/2) q[5];
cx q[7],q[6];
cx q[1],q[5];
cx q[6],q[7];
cx q[0],q[1];
cx q[2],q[6];
rz(pi/2) q[5];
cx q[1],q[0];
cx q[6],q[2];
rx(pi/2) q[5];
cx q[0],q[1];
cx q[2],q[6];
rz(pi/2) q[5];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cx q[1],q[5];
cx q[0],q[1];
rz(pi/2) q[5];
cx q[1],q[0];
rx(pi/2) q[5];
cx q[0],q[1];
rz(pi/2) q[5];
cx q[3],q[0];
rz(pi) q[5];
cx q[0],q[3];
rz(pi/2) q[5];
cx q[3],q[0];
rx(pi/2) q[5];
rz(pi/2) q[0];
rz(pi/2) q[5];
rx(pi/2) q[0];
cx q[5],q[4];
rz(pi/2) q[0];
cx q[4],q[2];
cx q[2],q[4];
cx q[4],q[2];
cx q[2],q[6];
rz(pi/2) q[4];
cx q[6],q[2];
rx(pi/2) q[4];
cx q[2],q[6];
rz(pi/2) q[4];
cx q[5],q[4];
rz(pi/2) q[6];
rz(pi/2) q[4];
rx(pi/2) q[6];
rx(pi/2) q[4];
rz(pi/2) q[6];
rz(pi/2) q[4];
cx q[7],q[6];
cx q[5],q[4];
rz(pi/2) q[6];
cx q[4],q[5];
rx(pi/2) q[6];
cx q[5],q[4];
rz(pi/2) q[6];
rz(pi/2) q[4];
cx q[6],q[7];
rx(pi/2) q[4];
cx q[7],q[6];
rz(pi/2) q[4];
cx q[6],q[7];
cx q[2],q[6];
cx q[7],q[8];
cx q[6],q[2];
cx q[8],q[7];
cx q[2],q[6];
cx q[7],q[8];
cx q[2],q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[2];
cx q[2],q[4];
cx q[4],q[2];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[1],q[5];
rz(pi/2) q[4];
cx q[5],q[1];
rx(pi/2) q[4];
cx q[1],q[5];
rz(pi/2) q[4];
cx q[1],q[0];
cx q[5],q[4];
rz(pi/2) q[0];
rz(3*pi/2) q[1];
rz(pi/2) q[4];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[4];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[4];
rz(pi/2) q[1];
rz(3*pi/2) q[4];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[2];
cx q[2],q[4];
cx q[4],q[2];
cx q[2],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
rz(pi/2) q[6];
rz(pi/2) q[7];
rx(pi/2) q[6];
rx(pi/2) q[7];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[2],q[6];
rz(pi/2) q[6];
rx(pi/2) q[6];
rz(pi/2) q[6];
cx q[2],q[6];
cx q[6],q[2];
cx q[2],q[6];
cx q[4],q[2];
cx q[6],q[7];
cx q[2],q[4];
rz(pi/2) q[7];
cx q[4],q[2];
rx(pi/2) q[7];
cx q[5],q[4];
rz(pi/2) q[7];
cx q[4],q[5];
cx q[6],q[7];
cx q[5],q[4];
cx q[7],q[6];
cx q[1],q[5];
cx q[6],q[7];
cx q[5],q[1];
rz(pi/2) q[6];
cx q[1],q[5];
rx(pi/2) q[6];
rz(pi/2) q[1];
rz(pi/2) q[6];
rx(pi/2) q[1];
cx q[2],q[6];
rz(pi/2) q[1];
rz(pi/2) q[6];
cx q[0],q[1];
rx(pi/2) q[6];
cx q[3],q[0];
rz(pi/2) q[1];
rz(pi/2) q[6];
cx q[0],q[3];
rx(pi/2) q[1];
rz(pi/2) q[6];
cx q[3],q[0];
rz(pi/2) q[1];
rx(pi/2) q[6];
rz(pi/2) q[1];
rz(pi/2) q[6];
rx(pi/2) q[1];
rz(pi/2) q[1];
cx q[5],q[1];
rz(pi/2) q[1];
cx q[5],q[4];
rx(pi/2) q[1];
cx q[4],q[5];
rz(pi/2) q[1];
cx q[5],q[4];
rz(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[1];
cx q[2],q[4];
rx(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[1];
cx q[2],q[6];
cx q[1],q[0];
rz(pi/2) q[6];
cx q[0],q[3];
cx q[1],q[5];
rx(pi/2) q[6];
cx q[3],q[0];
cx q[5],q[1];
rz(pi/2) q[6];
cx q[0],q[3];
cx q[1],q[5];
cx q[3],q[0];
cx q[5],q[4];
cx q[0],q[1];
cx q[4],q[5];
cx q[5],q[4];
cx q[1],q[5];
cx q[4],q[2];
cx q[5],q[1];
cx q[2],q[4];
cx q[1],q[5];
cx q[4],q[2];
cx q[2],q[6];
cx q[6],q[2];
cx q[2],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[2],q[6];
rz(pi/2) q[7];
cx q[6],q[2];
rx(pi/2) q[7];
cx q[2],q[6];
rz(pi/2) q[7];
cx q[4],q[2];
cx q[8],q[7];
cx q[2],q[4];
rz(pi/2) q[7];
cx q[4],q[2];
rx(pi/2) q[7];
cx q[5],q[4];
rz(pi/2) q[7];
cx q[4],q[5];
cx q[6],q[7];
cx q[5],q[4];
cx q[7],q[6];
cx q[1],q[5];
cx q[4],q[2];
cx q[6],q[7];
cx q[5],q[1];
cx q[2],q[4];
rz(pi/2) q[6];
cx q[7],q[8];
cx q[1],q[5];
cx q[4],q[2];
rx(pi/2) q[6];
cx q[8],q[7];
rz(pi/2) q[1];
rz(pi/2) q[5];
rz(pi/2) q[6];
cx q[7],q[8];
rx(pi/2) q[1];
cx q[2],q[6];
rx(pi/2) q[5];
rz(pi/2) q[8];
rz(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[5];
rz(pi/2) q[6];
rx(pi/2) q[8];
cx q[0],q[1];
cx q[2],q[4];
rx(pi/2) q[6];
rz(pi/2) q[8];
rz(pi/2) q[0];
rz(pi/2) q[1];
cx q[4],q[2];
rz(pi/2) q[6];
rx(pi/2) q[0];
rx(pi/2) q[1];
cx q[4],q[5];
cx q[6],q[7];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[5];
cx q[7],q[8];
rz(pi/2) q[1];
rx(pi/2) q[5];
cx q[6],q[7];
rz(pi/2) q[8];
rz(pi/2) q[1];
rz(pi/2) q[5];
cx q[7],q[6];
rx(pi/2) q[8];
rx(pi/2) q[1];
cx q[5],q[4];
cx q[6],q[7];
rz(pi/2) q[8];
rz(pi/2) q[1];
cx q[2],q[6];
cx q[4],q[5];
rz(pi) q[8];
cx q[6],q[2];
cx q[5],q[4];
rz(pi/2) q[8];
cx q[1],q[5];
cx q[2],q[6];
rz(pi/2) q[4];
rx(pi/2) q[8];
cx q[5],q[1];
rx(pi/2) q[4];
rz(pi/2) q[8];
cx q[1],q[5];
rz(pi/2) q[4];
cx q[1],q[0];
cx q[2],q[4];
rz(pi/2) q[0];
rz(pi) q[1];
rz(pi/2) q[4];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[4];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[4];
rz(pi/2) q[0];
rz(pi/2) q[1];
cx q[4],q[2];
rx(pi/2) q[0];
cx q[2],q[4];
rz(pi/2) q[0];
cx q[4],q[2];
cx q[3],q[0];
cx q[5],q[4];
rz(pi/2) q[0];
cx q[4],q[5];
rx(pi/2) q[0];
cx q[5],q[4];
rz(pi/2) q[0];
cx q[1],q[5];
rz(pi/2) q[0];
cx q[5],q[1];
rx(pi/2) q[0];
cx q[1],q[5];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(3*pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[0],q[3];
cx q[3],q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[5];
rx(pi/2) q[0];
cx q[5],q[1];
rz(pi/2) q[0];
cx q[1],q[5];
rz(pi/2) q[0];
cx q[5],q[4];
rx(pi/2) q[0];
cx q[4],q[5];
rz(pi/2) q[0];
cx q[5],q[4];
cx q[4],q[2];
cx q[2],q[4];
cx q[4],q[2];
cx q[2],q[6];
cx q[6],q[2];
cx q[2],q[6];
cx q[7],q[6];
rz(3*pi/2) q[6];
rz(3*pi/2) q[7];
rz(pi/2) q[6];
rz(pi/2) q[7];
rx(pi/2) q[6];
rx(pi/2) q[7];
rz(pi/2) q[6];
rz(pi/2) q[7];

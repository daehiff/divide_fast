OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[6];
rx(pi) q[8];
cx q[8],q[2];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
cx q[8],q[2];
cx q[8],q[1];
rz(pi/4) q[1];
cx q[8],q[1];
cx q[6],q[7];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[2],q[4];
rx(pi/2) q[2];
cx q[2],q[4];
cx q[7],q[4];
rz(3*pi/2) q[4];
cx q[7],q[4];
cx q[8],q[6];
cx q[6],q[0];
rz(3*pi/4) q[0];
cx q[6],q[0];
cx q[8],q[6];
rz(7*pi/4) q[0];
cx q[5],q[8];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[8];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[6],q[7];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[6];
rx(pi) q[8];
cx q[8],q[2];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
cx q[8],q[2];
cx q[8],q[1];
rz(pi/4) q[1];
cx q[8],q[1];
cx q[6],q[7];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[2],q[4];
rx(pi/2) q[2];
cx q[2],q[4];
cx q[7],q[4];
rz(3*pi/2) q[4];
cx q[7],q[4];
cx q[8],q[6];
cx q[6],q[0];
rz(3*pi/4) q[0];
cx q[6],q[0];
cx q[8],q[6];
rz(7*pi/4) q[0];
cx q[5],q[8];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[8];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[6],q[7];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[6];
rx(pi) q[8];
cx q[8],q[2];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
cx q[8],q[2];
cx q[8],q[1];
rz(pi/4) q[1];
cx q[8],q[1];
cx q[6],q[7];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[2],q[4];
rx(pi/2) q[2];
cx q[2],q[4];
cx q[7],q[4];
rz(3*pi/2) q[4];
cx q[7],q[4];
cx q[8],q[6];
cx q[6],q[0];
rz(3*pi/4) q[0];
cx q[6],q[0];
cx q[8],q[6];
rz(7*pi/4) q[0];
cx q[5],q[8];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[8];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[6],q[7];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[6];
rx(pi) q[8];
cx q[8],q[2];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
cx q[8],q[2];
cx q[8],q[1];
rz(pi/4) q[1];
cx q[8],q[1];
cx q[6],q[7];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[2],q[4];
rx(pi/2) q[2];
cx q[2],q[4];
cx q[7],q[4];
rz(3*pi/2) q[4];
cx q[7],q[4];
cx q[8],q[6];
cx q[6],q[0];
rz(3*pi/4) q[0];
cx q[6],q[0];
cx q[8],q[6];
rz(7*pi/4) q[0];
cx q[5],q[8];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[8];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[6],q[7];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[4];
rx(pi) q[6];
rx(pi) q[8];
cx q[8],q[2];
cx q[2],q[0];
rz(3*pi/2) q[0];
cx q[2],q[0];
cx q[8],q[2];
cx q[8],q[1];
rz(pi/4) q[1];
cx q[8],q[1];
cx q[6],q[7];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[2],q[4];
rx(pi/2) q[2];
cx q[2],q[4];
cx q[7],q[4];
rz(3*pi/2) q[4];
cx q[7],q[4];
cx q[8],q[6];
cx q[6],q[0];
rz(3*pi/4) q[0];
cx q[6],q[0];
cx q[8],q[6];
rz(7*pi/4) q[0];
cx q[5],q[8];
cx q[2],q[5];
rx(pi/4) q[2];
cx q[2],q[5];
cx q[5],q[8];
cx q[1],q[8];
rx(3*pi/2) q[1];
cx q[1],q[8];
cx q[6],q[7];

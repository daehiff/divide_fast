OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5],q[8];
cx q[8],q[4];
rz(5*pi/4) q[4];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[6];
rx(pi/2) q[4];
cx q[4],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[7],q[8];
cx q[8],q[1];
rz(3*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[7],q[8];
rx(5*pi/4) q[7];
cx q[7],q[8];
cx q[1],q[5];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[1],q[5];
rx(3*pi/2) q[3];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[6],q[8];
rx(5*pi/4) q[6];
cx q[6],q[8];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[4],q[1];
cx q[5],q[8];
cx q[8],q[4];
rz(5*pi/4) q[4];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[6];
rx(pi/2) q[4];
cx q[4],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[7],q[8];
cx q[8],q[1];
rz(3*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[7],q[8];
rx(5*pi/4) q[7];
cx q[7],q[8];
cx q[1],q[5];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[1],q[5];
rx(3*pi/2) q[3];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[6],q[8];
rx(5*pi/4) q[6];
cx q[6],q[8];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[4],q[1];
cx q[5],q[8];
cx q[8],q[4];
rz(5*pi/4) q[4];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[6];
rx(pi/2) q[4];
cx q[4],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[7],q[8];
cx q[8],q[1];
rz(3*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[7],q[8];
rx(5*pi/4) q[7];
cx q[7],q[8];
cx q[1],q[5];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[1],q[5];
rx(3*pi/2) q[3];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[6],q[8];
rx(5*pi/4) q[6];
cx q[6],q[8];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[4],q[1];
cx q[5],q[8];
cx q[8],q[4];
rz(5*pi/4) q[4];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[6];
rx(pi/2) q[4];
cx q[4],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[7],q[8];
cx q[8],q[1];
rz(3*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[7],q[8];
rx(5*pi/4) q[7];
cx q[7],q[8];
cx q[1],q[5];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[1],q[5];
rx(3*pi/2) q[3];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[6],q[8];
rx(5*pi/4) q[6];
cx q[6],q[8];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[4],q[1];
cx q[5],q[8];
cx q[8],q[4];
rz(5*pi/4) q[4];
cx q[8],q[4];
cx q[5],q[8];
cx q[4],q[6];
rx(pi/2) q[4];
cx q[4],q[6];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[7],q[8];
cx q[8],q[1];
rz(3*pi/4) q[1];
cx q[8],q[1];
cx q[7],q[8];
cx q[7],q[8];
rx(5*pi/4) q[7];
cx q[7],q[8];
cx q[1],q[5];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[1],q[5];
rx(3*pi/2) q[3];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[6],q[8];
rx(5*pi/4) q[6];
cx q[6],q[8];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[4],q[1];

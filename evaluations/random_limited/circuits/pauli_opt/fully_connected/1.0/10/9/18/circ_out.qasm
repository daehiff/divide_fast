OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2],q[3];
cx q[6],q[8];
cx q[7],q[1];
cx q[0],q[8];
cx q[2],q[5];
cx q[1],q[4];
cx q[7],q[4];
cx q[6],q[2];
cx q[8],q[5];
cx q[1],q[4];
cx q[1],q[7];
rx(pi) q[1];
cx q[1],q[7];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
rx(7*pi/4) q[8];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[8],q[2];
rz(pi/2) q[2];
cx q[8],q[2];
rz(3*pi/2) q[8];
cx q[6],q[0];
rz(7*pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[3];
cx q[1],q[4];
cx q[1],q[7];
rx(pi) q[1];
cx q[1],q[7];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
rx(7*pi/4) q[8];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[8],q[2];
rz(pi/2) q[2];
cx q[8],q[2];
rz(3*pi/2) q[8];
cx q[6],q[0];
rz(7*pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[3];
cx q[1],q[4];
cx q[1],q[7];
rx(pi) q[1];
cx q[1],q[7];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
rx(7*pi/4) q[8];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[8],q[2];
rz(pi/2) q[2];
cx q[8],q[2];
rz(3*pi/2) q[8];
cx q[6],q[0];
rz(7*pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[3];
cx q[1],q[4];
cx q[1],q[7];
rx(pi) q[1];
cx q[1],q[7];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
rx(7*pi/4) q[8];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[8],q[2];
rz(pi/2) q[2];
cx q[8],q[2];
rz(3*pi/2) q[8];
cx q[6],q[0];
rz(7*pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[3];
cx q[1],q[4];
cx q[1],q[7];
rx(pi) q[1];
cx q[1],q[7];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[3],q[8];
rx(pi/4) q[3];
cx q[3],q[8];
rx(7*pi/4) q[8];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[8],q[2];
rz(pi/2) q[2];
cx q[8],q[2];
rz(3*pi/2) q[8];
cx q[6],q[0];
rz(7*pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[3];
cx q[7],q[4];
cx q[6],q[2];
cx q[8],q[5];
cx q[0],q[8];
cx q[2],q[5];
cx q[1],q[4];
cx q[2],q[3];
cx q[6],q[8];
cx q[7],q[1];
